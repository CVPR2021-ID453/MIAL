import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmdet.models.dense_heads.anchor_head import MyEntLoss
from mmdet.core import encode_mask_results, tensor2imgs
from mmdet.models.detectors.base import *

# to use entropy
def get_rejection(scores, topk):
    # scores = torch.cat(scores, 0)
    rejection = (1 - scores).sum(1)
    arg = rejection.argsort()
    rejection = rejection[arg[-topk:]].mean(0, keepdim=True)

    return rejection

#
# def single_gpu_test_al(model,
#                        data_loader,
#                        show=False,
#                        out_dir=None,
#                        show_score_thr=0.3,
#                        return_box=False):
#     model.eval()
#     dataset = data_loader.dataset
#     rejections = torch.zeros(len(dataset)).cuda(torch.cuda.current_device())
#     print('>>> Computing Active Learning Metric...')
#     for i, data in enumerate(data_loader):
#         with torch.no_grad():
#             data['img'][0] = data['img'][0].cuda()
#             result = model(return_loss=False, rescale=True, return_box=return_box, **data)
#             result = torch.cat(result, 0)
#             rejections[i] = get_rejection(result, 1)
#             if i % 1000 == 0:
#                 print('>>> ', i, '/', len(dataset))
#

def single_gpu_test_al(model,
                       data_loader,
                       show=False,
                       out_dir=None,
                       show_score_thr=0.3,
                       return_box=False):
    model.eval()
    model.cuda()
    dataset = data_loader.dataset
    print('>>> Computing Active Learning Metric...')
    uncertainty = torch.zeros(len(dataset)).cuda(torch.cuda.current_device())

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data['img'][0] = data['img'][0].cuda()
            output1, output2, mil_score = model(return_loss=False, rescale=True, return_box=return_box, **data)
            cls_score1 = torch.cat(output1, 0)
            cls_score2 = torch.cat(output2, 0)
            cls_score1 = nn.Sigmoid()(cls_score1)
            cls_score2 = nn.Sigmoid()(cls_score2)

            loss_l2_p = (cls_score1 - cls_score2).pow(2)

            unc_anc = loss_l2_p.mean(dim=1)
            topk = 10000
            arg = unc_anc.argsort()
            unc = unc_anc[arg[-topk:]].mean()
            # unc = unc_anc.mean()

            # unc = loss_l2_p.mean(dim=1).mean()

            # ent1 = MyEntLoss().forward(output1[0])
            # ent2 = MyEntLoss().forward(output2[0])
            # ent_cat = torch.cat((torch.unsqueeze(ent1, 1), torch.unsqueeze(ent2, 1)), dim=1)
            # pred_loss = torch.mean(ent_cat, dim=1)
            # topk = 3
            # arg = pred_loss.argsort()
            # pred = pred_loss[arg[-topk:]].mean(0, keepdim=True)
            uncertainty[i] = unc
            if i % 1000 == 0:
                print('>>> ', i, '/', len(dataset))
    return uncertainty.cpu()

# def single_gpu_test_al(model,
#                        data_loader,
#                        show=False,
#                        out_dir=None,
#                        show_score_thr=0.3,
#                        return_box=False):
#     model.eval()
#     model.cuda()
#     dataset = data_loader.dataset
#     rejections = torch.zeros(len(dataset)).cuda(torch.cuda.current_device())
#     print('>>> Computing Active Learning Metric...')
#     uuncertainty = torch.tensor([]).cuda()
#
#     for i, data in enumerate(data_loader):
#         with torch.no_grad():
#             data['img'][0] = data['img'][0].cuda()
#             output1, output2 = model(return_loss=False, rescale=True, return_box=return_box, **data)
#             scores1 = torch.cat(output1, 0)
#             scores2 = torch.cat(output2, 0)
#
#             ent1 = MyEntLoss().forward(scores1)
#             ent2 = MyEntLoss().forward(scores2)
#             ent_cat = torch.cat((torch.unsqueeze(ent1, 1), torch.unsqueeze(ent2, 1)), dim=1)
#             pred_loss = torch.mean(ent_cat, dim=1)
#             topk = 1000
#             arg = pred_loss.argsort()
#             pred = pred_loss[arg[-topk:]].mean(0, keepdim=True)
#             uuncertainty = torch.cat((uuncertainty, pred), 0)
#
#             if i % 1000 == 0:
#                 print('>>> ', i, '/', len(dataset))
#
#     # return rejections
#     return uuncertainty.cpu()

def single_gpu_test_al_show(model,
                       data_loader,
                       show=False,
                       out_dir=None,
                       show_score_thr=0.3,
                       return_box=False):
    model.eval()
    model.cuda()
    dataset = data_loader.dataset
    rejections = torch.zeros(len(dataset)).cuda(torch.cuda.current_device())
    print('>>> Computing Active Learning Metric...')
    uuncertainty = torch.tensor([]).cuda()
    results1 = []
    results2 = []
    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data['img'][0] = data['img'][0].cuda()
            result1, result2 = model(return_loss=False, rescale=True, return_box=True, **data)
        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.show_result(
                    img_show,
                    result1,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

                model.show_result(
                    img_show,
                    result2,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result1, tuple):
            bbox_results, mask_results = result1
            encoded_mask_results = encode_mask_results(mask_results)
            result1 = bbox_results, encoded_mask_results
        results1.append(result1)

        if isinstance(result2, tuple):
            bbox_results, mask_results = result2
            encoded_mask_results = encode_mask_results(mask_results)
            result2 = bbox_results, encoded_mask_results
        results2.append(result2)

        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return results1, result2

    # return rejections
    return uuncertainty.cpu()
    # return rejections.cpu()




def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # result = model(return_loss=False, rescale=True, **data)
            result = model(return_loss=False, rescale=True, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)


        # encode mask results
        if isinstance(result, tuple):
            bbox_results, mask_results = result
            encoded_mask_results = encode_mask_results(mask_results)
            result = bbox_results, encoded_mask_results
        results.append(result)

        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return results


# def multi_gpu_test_al(model, data_loader, return_box=True, tmpdir='./tmp/', gpu_collect=True):
#     """Test model with multiple gpus.
#
#     This method tests model with multiple gpus and collects the results
#     under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
#     it encodes results to gpu tensors and use gpu communication for results
#     collection. On cpu mode it saves the results on different gpus to 'tmpdir'
#     and collects them by the rank 0 worker.
#
#     Args:
#         model (nn.Module): Model to be tested.
#         data_loader (nn.Dataloader): Pytorch data loader.
#         tmpdir (str): Path of directory to save the temporary results from
#             different gpus under cpu mode.
#         gpu_collect (bool): Option to use either gpu or cpu to collect results.
#
#     Returns:
#         list: The prediction results.
#     """
#     model.eval()
#     rejections = []
#     dataset = data_loader.dataset
#     time.sleep(2)  # This line can prevent deadlock problem in some cases.
#     for i, data in enumerate(data_loader):
#         with torch.no_grad():
#             data['img'][0] = data['img'][0].cuda(torch.cuda.current_device())
#             result = model(return_loss=False, rescale=True, return_box=return_box, **data)
#             result = torch.cat(result, 0)
#             rejection = get_rejection(result, 1)
#             if i % 1000 == 0:
#                 print('>>> ', i)
#         rejections.append(rejection)
#
#     # collect results from all ranks
#     print('done')
#     print(len(dataset))
#     if gpu_collect:
#         rejections = collect_results_gpu(rejections, len(dataset))
#     else:
#         rejections = collect_results_cpu(rejections, len(dataset), tmpdir)
#     return torch.cat(rejections, 0)


def multi_gpu_test_al(model, data_loader, return_box=True, tmpdir=None, gpu_collect=True):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data['img'][0] = data['img'][0].cuda(torch.cuda.current_device())
            result = model(return_loss=False, rescale=True, return_box=return_box, **data)
            result = torch.cat(result, 0)
            result = get_rejection(result, 1)
        results.append(result)

        if rank == 0:
            batch_size = (
                len(data['img_meta'].data)
                if 'img_meta' in data else len(data['img_metas'][0].data))
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result, tuple):
                bbox_results, mask_results = result
                encoded_mask_results = encode_mask_results(mask_results)
                result = bbox_results, encoded_mask_results
        results.append(result)

        if rank == 0:
            batch_size = (
                len(data['img_meta'].data)
                if 'img_meta' in data else len(data['img_metas'][0].data))
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
