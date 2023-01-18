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
import math
from typing import Iterable
import os
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import time
import base64
from io import BytesIO

import util.misc as misc
import util.lr_sched as lr_sched

from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.transforms import RotateAxisAngle


def evaluate_points(predicted_xyz, gt_xyz, dist_thres):
    if predicted_xyz.shape[0] == 0:
        return 0.0, 0.0, 0.0
    slice_size = 1000
    precision = 0.0
    for i in range(int(np.ceil(predicted_xyz.shape[0] / slice_size))):
        start = slice_size * i
        end   = slice_size * (i + 1)
        dist = ((predicted_xyz[start:end, None] - gt_xyz[None]) ** 2.0).sum(axis=-1) ** 0.5
        precision += ((dist < dist_thres).sum(axis=1) > 0).sum()
    precision /= predicted_xyz.shape[0]

    recall = 0.0
    for i in range(int(np.ceil(predicted_xyz.shape[0] / slice_size))):
        start = slice_size * i
        end   = slice_size * (i + 1)
        dist = ((predicted_xyz[:, None] - gt_xyz[None, start:end]) ** 2.0).sum(axis=-1) ** 0.5
        recall += ((dist < dist_thres).sum(axis=0) > 0).sum()
    recall /= gt_xyz.shape[0]
    return precision, recall, get_f1(precision, recall)

def aug_xyz(seen_xyz, unseen_xyz, args, is_train):
    degree_x = 0
    degree_y = 0
    degree_z = 0
    if is_train:
        r_delta = args.random_scale_delta
        scale = torch.tensor([
            random.uniform(1.0 - r_delta, 1.0 + r_delta),
            random.uniform(1.0 - r_delta, 1.0 + r_delta),
            random.uniform(1.0 - r_delta, 1.0 + r_delta),
        ], device=seen_xyz.device)

        if args.use_hypersim:
            shift = 0
        else:
            degree_x = random.randrange(-args.random_rotate_degree, args.random_rotate_degree + 1)
            degree_y = random.randrange(-args.random_rotate_degree, args.random_rotate_degree + 1)
            degree_z = random.randrange(-args.random_rotate_degree, args.random_rotate_degree + 1)

            r_shift = args.random_shift
            shift = torch.tensor([[[
                random.uniform(-r_shift, r_shift),
                random.uniform(-r_shift, r_shift),
                random.uniform(-r_shift, r_shift),
            ]]], device=seen_xyz.device)
        seen_xyz = seen_xyz * scale + shift
        unseen_xyz = unseen_xyz * scale + shift

    B, H, W, _ = seen_xyz.shape
    return [
        rotate(seen_xyz.reshape((B, -1, 3)), degree_x, degree_y, degree_z).reshape((B, H, W, 3)),
        rotate(unseen_xyz, degree_x, degree_y, degree_z),
    ]


def rotate(sample, degree_x, degree_y, degree_z):
    for degree, axis in [(degree_x, "X"), (degree_y, "Y"), (degree_z, "Z")]:
        if degree != 0:
            sample = RotateAxisAngle(degree, axis=axis).to(sample.device).transform_points(sample)
    return sample


def get_grid(B, device, co3d_world_size, granularity):
    N = int(np.ceil(2 * co3d_world_size / granularity))
    grid_unseen_xyz = torch.zeros((N, N, N, 3), device=device)
    for i in range(N):
        grid_unseen_xyz[i, :, :, 0] = i
    for j in range(N):
        grid_unseen_xyz[:, j, :, 1] = j
    for k in range(N):
        grid_unseen_xyz[:, :, k, 2] = k
    grid_unseen_xyz -= (N / 2.0)
    grid_unseen_xyz /= (N / 2.0) / co3d_world_size
    grid_unseen_xyz = grid_unseen_xyz.reshape((1, -1, 3)).repeat(B, 1, 1)
    return grid_unseen_xyz


def run_viz(model, data_loader, device, args, epoch):
    epoch_start_time = time.time()
    model.eval()
    os.system(f'mkdir {args.job_dir}/viz')

    print('Visualization data_loader length:', len(data_loader))
    dataset = data_loader.dataset
    for sample_idx, samples in enumerate(data_loader):
        if sample_idx >= args.max_n_viz_obj:
            break
        seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_images = prepare_data(samples, device, is_train=False, args=args, is_viz=True)

        pred_occupy = []
        pred_colors = []
        (model.module if hasattr(model, "module") else model).clear_cache()

        # don't forward all at once to avoid oom
        max_n_queries_fwd = 2000

        total_n_passes = int(np.ceil(unseen_xyz.shape[1] / max_n_queries_fwd))
        for p_idx in range(total_n_passes):
            p_start = p_idx     * max_n_queries_fwd
            p_end = (p_idx + 1) * max_n_queries_fwd
            cur_unseen_xyz = unseen_xyz[:, p_start:p_end]
            cur_unseen_rgb = unseen_rgb[:, p_start:p_end].zero_()
            cur_labels = labels[:, p_start:p_end].zero_()

            with torch.no_grad():
                _, pred, = model(
                    seen_images=seen_images,
                    seen_xyz=seen_xyz,
                    unseen_xyz=cur_unseen_xyz,
                    unseen_rgb=cur_unseen_rgb,
                    unseen_occupy=cur_labels,
                    cache_enc=args.run_viz,
                    valid_seen_xyz=valid_seen_xyz,
                )

            cur_occupy_out = pred[..., 0]

            if args.regress_color:
                cur_color_out = pred[..., 1:].reshape((-1, 3))
            else:
                cur_color_out = pred[..., 1:].reshape((-1, 3, 256)).max(dim=2)[1] / 255.0
            pred_occupy.append(cur_occupy_out)
            pred_colors.append(cur_color_out)

        rank = misc.get_rank()
        prefix = f'{args.job_dir}/viz/' + dataset.dataset_split + f'_ep{epoch}_rank{rank}_i{sample_idx}'

        img = (seen_images[0].permute(1, 2, 0) * 255).cpu().numpy().copy().astype(np.uint8)

        gt_xyz = samples[1][0].to(device).reshape(-1, 3)
        gt_rgb = samples[1][1].to(device).reshape(-1, 3)
        mesh_xyz = samples[2].to(device).reshape(-1, 3) if args.use_hypersim else None

        with open(prefix + '.html', 'a') as f:
            generate_html(
                img,
                seen_xyz, seen_images,
                torch.cat(pred_occupy, dim=1),
                torch.cat(pred_colors, dim=0),
                unseen_xyz,
                f,
                gt_xyz=gt_xyz,
                gt_rgb=gt_rgb,
                mesh_xyz=mesh_xyz,
            )
    print("Visualization epoch time:", time.time() - epoch_start_time)


def get_f1(precision, recall):
    if (precision + recall) == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)

def generate_html(img, seen_xyz, seen_rgb, pred_occ, pred_rgb, unseen_xyz, f,
        gt_xyz=None, gt_rgb=None, mesh_xyz=None, score_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
        pointcloud_marker_size=2,
    ):
    if img is not None:
        fig = plt.figure()
        plt.imshow(img)
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='jpg')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

        html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
        f.write(html)
        plt.close()

    clouds = {"MCC Output": {}}
    # Seen
    if seen_xyz is not None:
        seen_xyz = seen_xyz.reshape((-1, 3)).cpu()
        seen_rgb = torch.nn.functional.interpolate(seen_rgb, (112, 112)).permute(0, 2, 3, 1).reshape((-1, 3)).cpu()
        good_seen = seen_xyz[:, 0] != -100

        seen_pc = Pointclouds(
            points=seen_xyz[good_seen][None],
            features=seen_rgb[good_seen][None],
        )
        clouds["MCC Output"]["seen"] = seen_pc

    # GT points
    if gt_xyz is not None:
        subset_gt = random.sample(range(gt_xyz.shape[0]), 10000)
        gt_pc = Pointclouds(
            points=gt_xyz[subset_gt][None],
            features=gt_rgb[subset_gt][None],
        )
        clouds["MCC Output"]["GT points"] = gt_pc

    # GT meshes
    if mesh_xyz is not None:
        subset_mesh = random.sample(range(mesh_xyz.shape[0]), 10000)
        mesh_pc = Pointclouds(
            points=mesh_xyz[subset_mesh][None],
        )
        clouds["MCC Output"]["GT mesh"] = mesh_pc

    pred_occ = torch.nn.Sigmoid()(pred_occ).cpu()
    for t in score_thresholds:
        pos = pred_occ > t

        points = unseen_xyz[pos].reshape((-1, 3))
        features = pred_rgb[None][pos].reshape((-1, 3))
        good_points = points[:, 0] != -100

        if good_points.sum() == 0:
            continue

        pc = Pointclouds(
            points=points[good_points][None].cpu(),
            features=features[good_points][None].cpu(),
        )

        clouds["MCC Output"][f"pred_{t}"] = pc

    plt.figure()
    try:
        fig = plot_scene(clouds, pointcloud_marker_size=pointcloud_marker_size, pointcloud_max_points=20000 * 2)
        fig.update_layout(height=1000, width=1000)
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    except Exception as e:
        print('writing failed', e)
    try:
        plt.close()
    except:
        pass


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args=None):
    epoch_start_time = time.time()
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    print('Training data_loader length:', len(data_loader))
    for data_iter_step, samples in enumerate(data_loader):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_images = prepare_data(samples, device, is_train=True, args=args)

        with torch.cuda.amp.autocast():
            loss, _ = model(
                seen_images=seen_images,
                seen_xyz=seen_xyz,
                unseen_xyz=unseen_xyz,
                unseen_rgb=unseen_rgb,
                unseen_occupy=labels,
                valid_seen_xyz=valid_seen_xyz,
            )

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Warning: Loss is {}".format(loss_value))
            loss *= 0.0
            loss_value = 100.0

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    clip_grad=args.clip_grad,
                    update_grad=(data_iter_step + 1) % accum_iter == 0,
                    verbose=(data_iter_step % 100) == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if data_iter_step == 30:
            os.system('nvidia-smi')
            os.system('free -g')
        if args.debug and data_iter_step == 5:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print("Training epoch time:", time.time() - epoch_start_time)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_one_epoch(
        model: torch.nn.Module,
        data_loader: Iterable,
        device: torch.device,
        args=None
    ):
    epoch_start_time = time.time()
    model.train(False)

    metric_logger = misc.MetricLogger(delimiter="  ")

    print('Eval len(data_loader):', len(data_loader))

    for data_iter_step, samples in enumerate(data_loader):
        seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_images = prepare_data(samples, device, is_train=False, args=args)

        # don't forward all at once to avoid oom
        max_n_queries_fwd = 5000
        all_loss, all_preds = [], []
        for p_idx in range(int(np.ceil(unseen_xyz.shape[1] / max_n_queries_fwd))):
            p_start = p_idx     * max_n_queries_fwd
            p_end = (p_idx + 1) * max_n_queries_fwd
            cur_unseen_xyz = unseen_xyz[:, p_start:p_end]
            cur_unseen_rgb = unseen_rgb[:, p_start:p_end]
            cur_labels = labels[:, p_start:p_end]

            with torch.no_grad():
                loss, pred = model(
                    seen_images=seen_images,
                    seen_xyz=seen_xyz,
                    unseen_xyz=cur_unseen_xyz,
                    unseen_rgb=cur_unseen_rgb,
                    unseen_occupy=cur_labels,
                    valid_seen_xyz=valid_seen_xyz,
                )
            all_loss.append(loss)
            all_preds.append(pred)

        loss = sum(all_loss) / len(all_loss)
        pred = torch.cat(all_preds, dim=1)

        B = pred.shape[0]

        gt_xyz = samples[1][0].to(device).reshape((B, -1, 3))
        if args.use_hypersim:
            mesh_xyz = samples[2].to(device).reshape((B, -1, 3))

        s_thres = args.eval_score_threshold
        d_thres = args.eval_dist_threshold

        for b_idx in range(B):
            geometry_metrics = {}
            predicted_idx = torch.nn.Sigmoid()(pred[b_idx, :, 0]) > s_thres
            predicted_xyz = unseen_xyz[b_idx, predicted_idx]

            precision, recall, f1 = evaluate_points(predicted_xyz, gt_xyz[b_idx], d_thres)
            geometry_metrics[f'd{d_thres}_s{s_thres}_point_pr'] = precision
            geometry_metrics[f'd{d_thres}_s{s_thres}_point_rc'] = recall
            geometry_metrics[f'd{d_thres}_s{s_thres}_point_f1'] = f1

            if args.use_hypersim:
                precision, recall, f1 = evaluate_points(predicted_xyz, mesh_xyz[b_idx], d_thres)
                geometry_metrics[f'd{d_thres}_s{s_thres}_mesh_pr'] = precision
                geometry_metrics[f'd{d_thres}_s{s_thres}_mesh_rc'] = recall
                geometry_metrics[f'd{d_thres}_s{s_thres}_mesh_f1'] = f1

            metric_logger.update(**geometry_metrics)

        loss_value = loss.item()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        if args.debug and data_iter_step == 5:
            break

    metric_logger.synchronize_between_processes()
    print("Validation averaged stats:", metric_logger)
    print("Val epoch time:", time.time() - epoch_start_time)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def sample_uniform_semisphere(B, N, semisphere_size, device):
    for _ in range(100):
        points = torch.empty(B * N * 3, 3, device=device).uniform_(-semisphere_size, semisphere_size)
        points[..., 2] = points[..., 2].abs()
        dist = (points ** 2.0).sum(axis=-1) ** 0.5
        if (dist < semisphere_size).sum() >= B * N:
            return points[dist < semisphere_size][:B * N].reshape((B, N, 3))
        else:
            print('resampling sphere')


def get_grid_semisphere(B, granularity, semisphere_size, device):
    n_grid_pts = int(semisphere_size / granularity) * 2 + 1
    grid_unseen_xyz = torch.zeros((n_grid_pts, n_grid_pts, n_grid_pts // 2 + 1, 3), device=device)
    for i in range(n_grid_pts):
        grid_unseen_xyz[i, :, :, 0] = i
        grid_unseen_xyz[:, i, :, 1] = i
    for i in range(n_grid_pts // 2 + 1):
        grid_unseen_xyz[:, :, i, 2] = i
    grid_unseen_xyz[..., :2] -= (n_grid_pts // 2.0)
    grid_unseen_xyz *= granularity
    dist = (grid_unseen_xyz ** 2.0).sum(axis=-1) ** 0.5
    grid_unseen_xyz = grid_unseen_xyz[dist <= semisphere_size]
    return grid_unseen_xyz[None].repeat(B, 1, 1)


def get_min_dist(a, b, slice_size=1000):
    all_min, all_idx = [], []
    for i in range(int(np.ceil(a.shape[1] / slice_size))):
        start = slice_size * i
        end   = slice_size * (i + 1)
        # B, n_queries, n_gt
        dist = ((a[:, start:end] - b) ** 2.0).sum(axis=-1) ** 0.5
        # B, n_queries
        cur_min, cur_idx = dist.min(axis=2)
        all_min.append(cur_min)
        all_idx.append(cur_idx)
    return torch.cat(all_min, dim=1), torch.cat(all_idx, dim=1)


def construct_uniform_semisphere(gt_xyz, gt_rgb, semisphere_size, n_queries, dist_threshold, is_train, granularity):
    B = gt_xyz.shape[0]
    device = gt_xyz.device
    if is_train:
        unseen_xyz = sample_uniform_semisphere(B, n_queries, semisphere_size, device)
    else:
        unseen_xyz = get_grid_semisphere(B, granularity, semisphere_size, device)
    dist, idx_to_gt = get_min_dist(unseen_xyz[:, :, None], gt_xyz[:, None])
    labels = dist < dist_threshold
    unseen_rgb = torch.zeros_like(unseen_xyz)
    unseen_rgb[labels] = torch.gather(gt_rgb, 1, idx_to_gt.unsqueeze(-1).repeat(1, 1, 3))[labels]
    return unseen_xyz, unseen_rgb, labels.float()


def construct_uniform_grid(gt_xyz, gt_rgb, co3d_world_size, n_queries, dist_threshold, is_train, granularity):
    B = gt_xyz.shape[0]
    device = gt_xyz.device
    if is_train:
        unseen_xyz = torch.empty((B, n_queries, 3), device=device).uniform_(-co3d_world_size, co3d_world_size)
    else:
        unseen_xyz = get_grid(B, device, co3d_world_size, granularity)
    dist, idx_to_gt = get_min_dist(unseen_xyz[:, :, None], gt_xyz[:, None])
    labels = dist < dist_threshold
    unseen_rgb = torch.zeros_like(unseen_xyz)
    unseen_rgb[labels] = torch.gather(gt_rgb, 1, idx_to_gt.unsqueeze(-1).repeat(1, 1, 3))[labels]
    return unseen_xyz, unseen_rgb, labels.float()


def prepare_data(samples, device, is_train, args, is_viz=False):
    # Seen
    seen_xyz, seen_rgb = samples[0][0].to(device), samples[0][1].to(device)
    valid_seen_xyz = torch.isfinite(seen_xyz.sum(axis=-1))
    seen_xyz[~valid_seen_xyz] = -100
    B = seen_xyz.shape[0]
    # Gt
    gt_xyz, gt_rgb = samples[1][0].to(device).reshape(B, -1, 3), samples[1][1].to(device).reshape(B, -1, 3)

    sampling_func = construct_uniform_semisphere if args.use_hypersim else construct_uniform_grid
    unseen_xyz, unseen_rgb, labels = sampling_func(
        gt_xyz, gt_rgb,
        args.semisphere_size if args.use_hypersim else args.co3d_world_size,
        args.n_queries,
        args.train_dist_threshold,
        is_train,
        args.viz_granularity if is_viz else args.eval_granularity,
    )

    if is_train:
        seen_xyz, unseen_xyz = aug_xyz(seen_xyz, unseen_xyz, args, is_train=is_train)

        # Random Flip
        if random.random() < 0.5:
            seen_xyz[..., 0] *= -1
            unseen_xyz[..., 0] *= -1
            seen_xyz = torch.flip(seen_xyz, [2])
            valid_seen_xyz = torch.flip(valid_seen_xyz, [2])
            seen_rgb = torch.flip(seen_rgb, [3])

    return seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_rgb
