from mmdet.models.detectors.base import *


def calculate_uncertainty(cfg, model, data_loader, return_box=False):
    model.eval()
    model.cuda()
    dataset = data_loader.dataset
    print('>>> Computing Instance Uncertainty...')
    uncertainty = torch.zeros(len(dataset)).cuda(torch.cuda.current_device())
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data['img'][0] = data['img'][0].cuda()
            data.update({'x': data.pop('img')})
            y_head_f_1, y_head_f_2, y_head_f_mil = model(return_loss=False, rescale=True, return_box=return_box, **data)
            y_head_f_1 = torch.cat(y_head_f_1, 0)
            y_head_f_2 = torch.cat(y_head_f_2, 0)
            y_head_f_1 = nn.Sigmoid()(y_head_f_1)
            y_head_f_2 = nn.Sigmoid()(y_head_f_2)
            loss_l2_p = (y_head_f_1 - y_head_f_2).pow(2)
            uncertainty_all_N = loss_l2_p.mean(dim=1)
            arg = uncertainty_all_N.argsort()
            uncertainty_single = uncertainty_all_N[arg[-cfg.k:]].mean()
            uncertainty[i] = uncertainty_single
            if i % 1000 == 0:
                print('>>> ', i, '/', len(dataset))
    return uncertainty.cpu()


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    y_heads = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data.update({'x': data.pop('img')})
            y_head = model(return_loss=False, rescale=True, **data)
        y_heads.append(y_head)
        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return y_heads
