# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os
import sys
sys.path.append('../')

import torch
from util.hypersim_utils import read_h5py, read_img


def load_scene_names(hypersim_path, is_train):
    split = 'train' if is_train else 'test'
    scene_names = []
    with open(os.path.join(
            hypersim_path,
            'evermotion_dataset/analysis/metadata_images_split_scene_v1.csv'),'r') as f:
        for line in f:
            items = line.split(',')
            if items[-1].strip() == split:
                scene_names.append(items[0])
    scene_names = sorted(list(set(scene_names)))
    print(len(scene_names), 'scenes loaded:', scene_names)
    return scene_names


def main(args):
    for is_train in [False, True]:
        gt = {}
        for scene_name in load_scene_names(hypersim_path=args.hypersim_path, is_train=is_train):
            print('loading GT', scene_name)
            all_xyz = []
            all_rgb = []
            for frame_path in glob.glob(os.path.join(args.hypersim_path, scene_name, 'images/scene_cam_*_final_preview/*tonemap*')):
                frame_xyz_path = frame_path.replace('final_preview/', 'geometry_hdf5/').replace('.tonemap.jpg', '.position.hdf5')
                xyz = read_h5py(frame_xyz_path)
                img = read_img(frame_path)
                all_xyz.append(torch.nn.functional.interpolate(
                        xyz[None].permute(0, 3, 1, 2), (112, 112),
                        mode='bilinear',
                    )[0])
                all_rgb.append(torch.nn.functional.interpolate(
                        img[None].permute(0, 3, 1, 2), (112, 112),
                        mode='bilinear',
                    )[0])
            all_xyz = torch.cat(all_xyz, dim=1).reshape((3, -1)).permute(1, 0)
            all_rgb = torch.cat(all_rgb, dim=1).reshape((3, -1)).permute(1, 0)
            gt[scene_name] = (all_xyz, all_rgb)

        dataset_split = 'train' if is_train else 'val'
        torch.save(gt, f'hypersim_gt_{dataset_split}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MCC', add_help=False)
    parser.add_argument('--hypersim_path', default='', type=str, help='path to hypersim dataset')
    args = parser.parse_args()
    main(args)