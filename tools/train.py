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
from mmdet.apis import single_gpu_test, calculate_uncertainty
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.datasets import build_dataloader, build_dataset
from tools.utils import losstype
from mmdet.utils.active_datasets import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_directory', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_false',
                        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int,
                            help='number of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+',
                            help='ids of gpus to use (only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
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
    # work_directory is determined in this priority: CLI > segment in file > filename
    if args.work_directory is not None:
        # update configs according to CLI args if args.work_directory is not None
        cfg.work_directory = args.work_directory
    elif cfg.get('work_directory', None) is None:
        # use config filename as default work_directory if cfg.work_directory is None
        cfg.work_directory = osp.join('./work_directory/retina', osp.splitext(osp.basename(args.config))[0])
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
    # create work_directory
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_directory))
    # dump config
    cfg.dump(osp.join(cfg.work_directory, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_directory, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    # init the meta dict to record some important information such as environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # ---------- MIAL Training and Test Start Here ---------- #

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    X_L, X_U, X_all, all_anns = get_X_L_0(cfg)

    # # load set and model
    # # Please change it to the timestamp directory which you want to load data from.
    # last_timestamp = '/20201013_154728'
    # # Please change it to the cycle which you want to load data from.
    # load_cycle = 0
    # X_L = np.load(cfg.work_directory + last_timestamp +'/X_L_' + str(load_cycle) + '.npy')
    # X_U = np.load(cfg.work_directory + last_timestamp +'/X_U_' + str(load_cycle) + '.npy')
    # cfg.cycles = list(range(load_cycle, 7))

    cfg.work_directory = cfg.work_directory + '/' + timestamp
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_directory))
    np.save(cfg.work_directory + '/X_L_' + '0' + '.npy', X_L)
    np.save(cfg.work_directory + '/X_U_' + '0' + '.npy', X_U)
    initial_step = cfg.lr_config.step
    theta_f_1 = ['bbox_head.f_1_convs.0.conv.weight', 'bbox_head.f_1_convs.0.conv.bias',
                 'bbox_head.f_1_convs.1.conv.weight', 'bbox_head.f_1_convs.1.conv.bias',
                 'bbox_head.f_1_convs.2.conv.weight', 'bbox_head.f_1_convs.2.conv.bias',
                 'bbox_head.f_1_convs.3.conv.weight', 'bbox_head.f_1_convs.3.conv.bias',
                 'bbox_head.f_1_retina.weight', 'bbox_head.f_1_retina.bias']
    theta_f_2 = ['bbox_head.f_2_convs.0.conv.weight', 'bbox_head.f_2_convs.0.conv.bias',
                 'bbox_head.f_2_convs.1.conv.weight', 'bbox_head.f_2_convs.1.conv.bias',
                 'bbox_head.f_2_convs.2.conv.weight', 'bbox_head.f_2_convs.2.conv.bias',
                 'bbox_head.f_2_convs.3.conv.weight', 'bbox_head.f_2_convs.3.conv.bias',
                 'bbox_head.f_2_retina.weight', 'bbox_head.f_2_retina.bias']
    for cycle in cfg.cycles:
        # set random seeds
        if args.seed is not None:
            logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
            set_random_seed(args.seed, deterministic=args.deterministic)
        cfg.seed = args.seed
        meta['seed'] = args.seed
        # get the config of the labeled dataset
        cfg = create_X_L_file(cfg, X_L, all_anns, cycle)
        # load model
        model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

        # # Please change it to the epoch which you want to load model at.
        # model_file_name = '/latest.pth'
        # model.load_state_dict(torch.load(cfg.work_directory[:16] + last_timestamp + model_file_name)['state_dict'])

        # load dataset
        datasets = [build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))
        if cfg.checkpoint_config is not None and cycle == 0:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(mmdet_version=__version__ + get_git_hash()[:7],
                                              config=cfg.pretty_text, CLASSES=datasets[0].CLASSES)
        model.CLASSES = datasets[0].CLASSES
        for epoch in range(cfg.epoch):
            # Only in the last 3 epoch does the learning rate need to be reduced and the model needs to be evaluated.
            if epoch == cfg.epoch - 1:
                cfg.lr_config.step = initial_step
                cfg.evaluation.interval = cfg.epoch_ratio[0]
            else:
                cfg.lr_config.step = [1000]
                cfg.evaluation.interval = 100

            # ---------- Label Set Training ----------

            if epoch == 0:
                cfg = create_X_L_file(cfg, X_L, all_anns, cycle)
                datasets = [build_dataset(cfg.data.train)]
                losstype.update_vars(0)
                cfg.total_epochs = cfg.epoch_ratio[0]
                cfg_bak = cfg.deepcopy()
                time.sleep(2)
                for name, value in model.named_parameters():
                    value.requires_grad = True
                train_detector(model, datasets, cfg,
                               distributed=distributed, validate=(not args.no_validate), timestamp=timestamp, meta=meta)
                cfg = cfg_bak

            # ---------- Re-weighting and Minimizing Instance Uncertainty ----------

            cfg_u = create_X_U_file(cfg.deepcopy(), X_U, all_anns, cycle)
            cfg = create_X_L_file(cfg, X_L, all_anns, cycle)
            datasets_u = [build_dataset(cfg_u.data.train)]
            datasets = [build_dataset(cfg.data.train)]
            losstype.update_vars(1)
            cfg_u.total_epochs = cfg_u.epoch_ratio[1]
            cfg.total_epochs = cfg.epoch_ratio[1]
            cfg_u_bak = cfg_u.deepcopy()
            cfg_bak = cfg.deepcopy()
            time.sleep(2)
            for name, value in model.named_parameters():
                if name in theta_f_1:
                    value.requires_grad = False
                elif name in theta_f_2:
                    value.requires_grad = False
                else:
                    value.requires_grad = True
            train_detector(model, [datasets, datasets_u], [cfg, cfg_u],
                           distributed=distributed, validate=(not args.no_validate), timestamp=timestamp, meta=meta)
            cfg_u = cfg_u_bak
            cfg = cfg_bak

            # ---------- Re-weighting and Maximizing Instance Uncertainty ----------

            cfg_u = create_X_U_file(cfg.deepcopy(), X_U, all_anns, cycle)
            cfg = create_X_L_file(cfg, X_L, all_anns, cycle)
            datasets_u = [build_dataset(cfg_u.data.train)]
            datasets = [build_dataset(cfg.data.train)]
            losstype.update_vars(2)
            cfg_u.total_epochs = cfg_u.epoch_ratio[1]
            cfg.total_epochs = cfg.epoch_ratio[1]
            cfg_u_bak = cfg_u.deepcopy()
            cfg_bak = cfg.deepcopy()
            time.sleep(2)
            for name, value in model.named_parameters():
                if name in theta_f_1:
                    value.requires_grad = True
                elif name in theta_f_2:
                    value.requires_grad = True
                else:
                    value.requires_grad = False
            train_detector(model, [datasets, datasets_u], [cfg, cfg_u],
                           distributed=distributed,validate=(not args.no_validate), timestamp=timestamp, meta=meta)
            cfg_u = cfg_u_bak
            cfg = cfg_bak

            # ---------- Label Set Training ----------

            cfg = create_X_L_file(cfg, X_L, all_anns, cycle)
            datasets = [build_dataset(cfg.data.train)]
            losstype.update_vars(0)
            cfg.total_epochs = cfg.epoch_ratio[0]
            cfg_bak = cfg.deepcopy()
            for name, value in model.named_parameters():
                value.requires_grad = True
            time.sleep(2)
            train_detector(model, datasets, cfg,
                           distributed=distributed, validate=args.no_validate, timestamp=timestamp, meta=meta)
            cfg = cfg_bak

        # ---------- Informative Image Selection ----------

        if cycle != cfg.cycles[-1]:
            # get new labeled data
            dataset_al = build_dataset(cfg.data.test)
            data_loader = build_dataloader(dataset_al, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
                                           dist=False, shuffle=False)
            # set random seeds
            if args.seed is not None:
                logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
                set_random_seed(args.seed, deterministic=args.deterministic)
            cfg.seed = args.seed
            meta['seed'] = args.seed
            uncertainty = calculate_uncertainty(cfg, model, data_loader, return_box=False)
            # update labeled set
            X_L, X_U = update_X_L(uncertainty, X_all, X_L, cfg.X_S_size)
            # save set and model
            np.save(cfg.work_directory + '/X_L_' + str(cycle+1) + '.npy', X_L)
            np.save(cfg.work_directory + '/X_U_' + str(cycle+1) + '.npy', X_U)


if __name__ == '__main__':
    main()
