# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import timm.optim.optim_factory as optim_factory

import util.misc as misc
import mcc_model
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.hypersim_dataset import HyperSimDataset, hypersim_collate_fn
from util.co3d_dataset import CO3DV2Dataset, co3dv2_collate_fn
from engine_mcc import train_one_epoch, run_viz, eval_one_epoch
from util.co3d_utils import get_all_dataset_maps


def get_args_parser():
    parser = argparse.ArgumentParser('MCC', add_help=False)

    # Model
    parser.add_argument('--input_size', default=224, type=int,
                        help='Images input size')
    parser.add_argument('--occupancy_weight', default=1.0, type=float,
                        help='A constant to weight the occupancy loss')
    parser.add_argument('--rgb_weight', default=0.01, type=float,
                        help='A constant to weight the color prediction loss')
    parser.add_argument('--n_queries', default=550, type=int,
                        help='Number of queries used in decoder.')
    parser.add_argument('--drop_path', default=0.1, type=float,
                        help='drop_path probability')
    parser.add_argument('--regress_color', action='store_true',
                        help='If true, regress color with MSE. Otherwise, 256-way classification for each channel.')

    # Training
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU for training (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--eval_batch_size', default=2, type=int,
                        help='Batch size per GPU for evaluation (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 512')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warmup LR')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Clip gradient at the specified norm')

    # Job
    parser.add_argument('--job_dir', default='',
                        help='Path to where to save, empty for no saving')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Path to where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed.')
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Start epoch')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers for training data loader')
    parser.add_argument('--num_eval_workers', default=4, type=int,
                        help='Number of workers for evaluation data loader')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='Url used to set up distributed training')

    # Experiments
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--run_viz', action='store_true',
                        help='Specify to run only the visualization/inference given a trained model.')
    parser.add_argument('--max_n_viz_obj', default=64, type=int,
                        help='Max number of objects to visualize during training.')

    # Data
    parser.add_argument('--train_epoch_len_multiplier', default=32, type=int,
                        help='# examples per training epoch is # objects * train_epoch_len_multiplier')
    parser.add_argument('--eval_epoch_len_multiplier', default=1, type=int,
                        help='# examples per eval epoch is # objects * eval_epoch_len_multiplier')

    # CO3D
    parser.add_argument('--co3d_path', type=str, default='co3d_data',
                        help='Path to CO3D v2 data.')
    parser.add_argument('--holdout_categories', action='store_true',
                        help='If true, hold out 10 categories and train on only the remaining 41 categories.')
    parser.add_argument('--co3d_world_size', default=3.0, type=float,
                        help='The world space we consider is \in [-co3d_world_size, co3d_world_size] in each dimension.')

    # Hypersim
    parser.add_argument('--use_hypersim', action='store_true',
                        help='If true, use hypersim, else, co3d.')
    parser.add_argument('--hypersim_path', default="hypersim_data", type=str,
                        help="Path to Hypersim data.")

    # Data aug
    parser.add_argument('--random_scale_delta', default=0.2, type=float,
                        help='Random scaling each example by a scaler \in [1 - random_scale_delta, 1 + random_scale_delta].')
    parser.add_argument('--random_shift', default=1.0, type=float,
                        help='Random shifting an example in each axis by an amount \in [-random_shift, random_shift]')
    parser.add_argument('--random_rotate_degree', default=180, type=int,
                        help='Random rotation degrees.')

    # Smapling, evaluation, and coordinate system
    parser.add_argument('--shrink_threshold', default=10.0, type=float,
                        help='Any points with distance beyond this value will be shrunk.')
    parser.add_argument('--semisphere_size', default=6.0, type=float,
                        help='The Hypersim task predicts points in a semisphere in front of the camera.'
                             'This value specifies the size of the semisphere.')
    parser.add_argument('--eval_granularity', default=0.1, type=float,
                        help='Granularity of the evaluation points.')
    parser.add_argument('--viz_granularity', default=0.1, type=float,
                        help='Granularity of points in visaulizatoin.')

    parser.add_argument('--eval_score_threshold', default=0.1, type=float,
                        help='Score threshold for evaluation.')
    parser.add_argument('--eval_dist_threshold', default=0.1, type=float,
                        help='Points closer than this amount to a groud-truth is considered correct.')
    parser.add_argument('--train_dist_threshold', default=0.1, type=float,
                        help='Points closer than this amount is considered positive in training.')
    return parser


def build_loader(args, num_tasks, global_rank, is_train, dataset_type, collate_fn, dataset_maps):
    '''Build data loader'''
    dataset = dataset_type(args, is_train=is_train, dataset_maps=dataset_maps)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size if is_train else args.eval_batch_size,
        sampler=sampler_train,
        num_workers=args.num_workers if is_train else args.num_eval_workers,
        pin_memory=args.pin_mem,
        collate_fn=collate_fn,
    )
    return data_loader


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # define the model
    model = mcc_model.get_mcc_model(
        rgb_weight=args.rgb_weight,
        occupancy_weight=args.occupancy_weight,
        args=args,
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 512

    print("base lr: %.2e" % (args.blr))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.use_hypersim:
        dataset_type = HyperSimDataset
        collate_fn = hypersim_collate_fn
        dataset_maps = None
    else:
        dataset_type = CO3DV2Dataset
        collate_fn = co3dv2_collate_fn
        dataset_maps = get_all_dataset_maps(
            args.co3d_path, args.holdout_categories,
        )

    dataset_viz = dataset_type(args, is_train=False, is_viz=True, dataset_maps=dataset_maps)
    sampler_viz = torch.utils.data.DistributedSampler(
        dataset_viz, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )

    data_loader_viz = torch.utils.data.DataLoader(
        dataset_viz, batch_size=1,
        sampler=sampler_viz,
        num_workers=args.num_eval_workers,
        pin_memory=args.pin_mem,
        collate_fn=collate_fn,
    )

    if args.run_viz:
        run_viz(
            model, data_loader_viz,
            device, args=args, epoch=0,
        )
        exit()

    data_loader_train, data_loader_val = [
        build_loader(
            args, num_tasks, global_rank,
            is_train=is_train,
            dataset_type=dataset_type, collate_fn=collate_fn, dataset_maps=dataset_maps
        ) for is_train in [True, False]
    ]

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        print(f'Epoch {epoch}:')
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args=args,
        )

        val_stats = {}
        if (epoch % 5 == 4 or epoch + 1 == args.epochs) or args.debug:
            val_stats = eval_one_epoch(
                model, data_loader_val,
                device, args=args,
            )

        if ((epoch % 10 == 9 or epoch + 1 == args.epochs) or args.debug):
            run_viz(
                model, data_loader_viz,
                device, args=args, epoch=epoch,
            )

        if args.output_dir and (epoch % 10 == 9 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    run_viz(
        model, data_loader_viz,
        device, args=args, epoch=-1,
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

