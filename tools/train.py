import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
import random
import numpy as np
from mmcv import Config, DictAction
from mmcv.runner import init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.apis import multi_gpu_test, single_gpu_test, multi_gpu_test_al, single_gpu_test_al
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.datasets import build_dataloader, build_dataset
from tools.utils import losstype
from mmdet.utils.active_datasets import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_false',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs/ALDet/SSD',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dira
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    labeled_set, unlabeled_set, all_set, all_anns = get_init_labeled_set(cfg)

    # # load set and model
    # last_timestamp = '/20201013_154728'
    # load_cycle = 0
    # labeled_set = np.load(cfg.work_dir + last_timestamp +'/L_cycle_' + str(load_cycle) + '.npy')
    # unlabeled_set = np.load(cfg.work_dir + last_timestamp +'/U_cycle_' + str(load_cycle) + '.npy')
    # cfg.active_learning.cycles = [7]
    # # model.load_state_dict(torch.load(cfg.work_dir[:18] + last_timestamp + '/latest' + '.pth')['state_dict'])

    cfg.work_dir = cfg.work_dir + '/' + timestamp
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    np.save(cfg.work_dir + '/L_cycle_' + '0' + '.npy', labeled_set)
    np.save(cfg.work_dir + '/U_cycle_' + '0' + '.npy', unlabeled_set)
    initial_step = cfg.lr_config.step
    f1_name_list = ['bbox_head.cls_convs1.0.conv.weight', 'bbox_head.cls_convs1.0.conv.bias',
                    'bbox_head.cls_convs1.1.conv.weight', 'bbox_head.cls_convs1.1.conv.bias',
                    'bbox_head.cls_convs1.2.conv.weight', 'bbox_head.cls_convs1.2.conv.bias',
                    'bbox_head.cls_convs1.3.conv.weight', 'bbox_head.cls_convs1.3.conv.bias',
                    'bbox_head.retina_cls1.weight', 'bbox_head.retina_cls1.bias']
    f2_name_list = ['bbox_head.cls_convs2.0.conv.weight', 'bbox_head.cls_convs2.0.conv.bias',
                    'bbox_head.cls_convs2.1.conv.weight', 'bbox_head.cls_convs2.1.conv.bias',
                    'bbox_head.cls_convs2.2.conv.weight', 'bbox_head.cls_convs2.2.conv.bias',
                    'bbox_head.cls_convs2.3.conv.weight', 'bbox_head.cls_convs2.3.conv.bias',
                    'bbox_head.retina_cls2.weight', 'bbox_head.retina_cls2.bias']

    for cycle in cfg.active_learning.cycles:

        # set random seeds
        if args.seed is not None:
            logger.info(f'Set random seed to {args.seed}, '
                        f'deterministic: {args.deterministic}')
            set_random_seed(args.seed, deterministic=args.deterministic)
        cfg.seed = args.seed
        meta['seed'] = args.seed

        # get labeled dataset cfg
        cfg = create_active_labeled_set(cfg, labeled_set, all_anns, cycle)

        # load model
        model = build_detector(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        # model.load_state_dict(torch.load(cfg.work_dir[:18] + last_timestamp + '/epoch_3' + '.pth')['state_dict'])
        # load dataset
        datasets = [build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))

        if cfg.checkpoint_config is not None and cycle == 0:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__ + get_git_hash()[:7],
                config=cfg.pretty_text,
                CLASSES=datasets[0].CLASSES)

        # add an attribute for visualization convenience
        # cfg_bak = cfg.deepcopy()
        model.CLASSES = datasets[0].CLASSES

        # add an attribute for visualization convenience
        for epoch in range(cfg.active_learning.epoch):

            if epoch == cfg.active_learning.epoch - 1:
                cfg.lr_config.step = initial_step
                cfg.evaluation.interval = cfg.epoch_minmax[0]
            else:
                cfg.lr_config.step = [1000]
                cfg.evaluation.interval = 100

            model.CLASSES = datasets[0].CLASSES

            # initial
            if epoch == 0:
                cfg = create_active_labeled_set(cfg, labeled_set, all_anns, cycle)
                datasets = [build_dataset(cfg.data.train)]
                losstype.update_vars(0)
                cfg.total_epochs = cfg.epoch_minmax[0]
                cfg_bak = cfg.deepcopy()
                time.sleep(2)
                for name, value in model.named_parameters():
                    value.requires_grad = True
                train_detector(
                    model,
                    datasets,
                    cfg,
                    distributed=distributed,
                    validate=(not args.no_validate),
                    timestamp=timestamp,
                    meta=meta)
                cfg = cfg_bak

            # agreement
            cfg_u = create_active_unlabeled_set(cfg.deepcopy(), unlabeled_set, all_anns, cycle)
            cfg = create_active_labeled_set(cfg, labeled_set, all_anns, cycle)
            datasets_u = [build_dataset(cfg_u.data.train)]
            datasets = [build_dataset(cfg.data.train)]
            losstype.update_vars(1)
            cfg_u.total_epochs = cfg_u.epoch_minmax[1]
            cfg.total_epochs = cfg.epoch_minmax[1]
            cfg_u_bak = cfg_u.deepcopy()
            cfg_bak = cfg.deepcopy()
            time.sleep(2)
            for name, value in model.named_parameters():
                if name in f1_name_list:
                    value.requires_grad = False
                elif name in f2_name_list:
                    value.requires_grad = False
                else:
                    value.requires_grad = True
            train_detector(
                model,
                [datasets, datasets_u],
                [cfg, cfg_u],
                distributed=distributed,
                validate=(not args.no_validate),
                timestamp=timestamp,
                meta=meta)
            cfg_u = cfg_u_bak
            cfg = cfg_bak

            # discrepancy
            cfg_u = create_active_unlabeled_set(cfg.deepcopy(), unlabeled_set, all_anns, cycle)
            cfg = create_active_labeled_set(cfg, labeled_set, all_anns, cycle)
            datasets_u = [build_dataset(cfg_u.data.train)]
            datasets = [build_dataset(cfg.data.train)]
            losstype.update_vars(2)
            cfg_u.total_epochs = cfg_u.epoch_minmax[1]
            cfg.total_epochs = cfg.epoch_minmax[1]
            cfg_u_bak = cfg_u.deepcopy()
            cfg_bak = cfg.deepcopy()
            time.sleep(2)
            for name, value in model.named_parameters():
                if name in f1_name_list:
                    value.requires_grad = True
                elif name in f2_name_list:
                    value.requires_grad = True
                else:
                    value.requires_grad = False
            train_detector(
                model,
                [datasets, datasets_u],
                [cfg, cfg_u],
                distributed=distributed,
                validate=(not args.no_validate),
                timestamp=timestamp,
                meta=meta)
            cfg_u = cfg_u_bak
            cfg = cfg_bak

            # initial
            cfg = create_active_labeled_set(cfg, labeled_set, all_anns, cycle)
            datasets = [build_dataset(cfg.data.train)]
            losstype.update_vars(0)
            cfg.total_epochs = cfg.epoch_minmax[0]
            cfg_bak = cfg.deepcopy()
            for name, value in model.named_parameters():
                value.requires_grad = True
            time.sleep(2)
            train_detector(
                model,
                datasets,
                cfg,
                distributed=distributed,
                validate=args.no_validate,
                timestamp=timestamp,
                meta=meta)
            cfg = cfg_bak

        if cycle != cfg.active_learning.cycles[-1]:

            # get new labeled data
            dataset_al = build_dataset(cfg.data.test)
            data_loader = build_dataloader(
                dataset_al,
                samples_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=False,
                shuffle=False)

            # active_metric = single_gpu_test_al(model, data_loader, return_box=False)
            # set random seeds
            if args.seed is not None:
                logger.info(f'Set random seed to {args.seed}, '
                            f'deterministic: {args.deterministic}')
                set_random_seed(args.seed, deterministic=args.deterministic)
            cfg.seed = args.seed
            meta['seed'] = args.seed
            active_metric = single_gpu_test_al(model, data_loader, return_box=False)
            # active_metric = torch.rand(16551)
            # active_metric = active_metric.data.cpu()
            # update labeled set
            labeled_set, unlabeled_set = update_labeled_set(
                active_metric,
                all_set,
                labeled_set,
                cfg.active_learning.budget
            )

            # save set and model
            np.save(cfg.work_dir + '/L_cycle_' + str(cycle+1) + '.npy', labeled_set)
            np.save(cfg.work_dir + '/U_cycle_' + str(cycle+1) + '.npy', unlabeled_set)


if __name__ == '__main__':
    main()
