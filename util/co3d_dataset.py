# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import cast

import torch
from pytorch3d.implicitron.dataset.dataset_base import FrameData

import util.co3d_utils as co3d_utils


def co3dv2_collate_fn(batch):
    assert len(batch[0]) == 4
    return (
        FrameData.collate([x[0] for x in batch]),
        FrameData.collate([x[1] for x in batch]),
        [x[2] for x in batch],
        [x[3] for x in batch],
    )


def pad_point_cloud(pc, N):
    cur_N = pc._points_list[0].shape[0]
    if cur_N == N:
        return pc

    assert cur_N > 0

    n_pad = N - cur_N
    indices = random.choices(list(range(cur_N)), k=n_pad)
    pc._features_list[0] = torch.cat([pc._features_list[0], pc._features_list[0][indices]], dim=0)
    pc._points_list[0] = torch.cat([pc._points_list[0], pc._points_list[0][indices]], dim=0)
    return pc


class CO3DV2Dataset(torch.utils.data.Dataset):
    def __init__(self, args, is_train, is_viz=False, dataset_maps=None):

        self.args = args
        self.is_train = is_train
        self.is_viz = is_viz

        self.dataset_split = 'train' if is_train else 'val'
        self.all_datasets = dataset_maps[0 if is_train else 1]
        print(len(self.all_datasets), 'categories loaded')

        self.all_example_names = self.get_all_example_names()
        print('containing', len(self.all_example_names), 'examples')

    def get_all_example_names(self):
        all_example_names = []
        for category in self.all_datasets.keys():
            for sequence_name in self.all_datasets[category].seq_name2idx.keys():
                all_example_names.append((category, sequence_name))
        return all_example_names

    def __getitem__(self, index):
        for retry in range(1000):
            try:
                if retry > 9:
                    index = random.choice(range(len(self)))
                    print('retry', retry, 'new index:', index)
                gap = 1 if self.is_train else len(self.all_example_names) // len(self)
                assert gap >= 1
                category, sequence_name = self.all_example_names[(index * gap) % len(self.all_example_names)]

                cat_dataset = self.all_datasets[category]

                frame_data = cat_dataset.__getitem__(
                    random.choice(cat_dataset.seq_name2idx[sequence_name])
                    if self.is_train
                    else cat_dataset.seq_name2idx[sequence_name][
                        hash(sequence_name) % len(cat_dataset.seq_name2idx[sequence_name])
                    ]
                )
                test_frame = None
                seen_idx = None

                frame_data = cat_dataset.frame_data_type.collate([frame_data])
                mask = (
                    (cast(torch.Tensor, frame_data.fg_probability) > 0.5).float()
                    if frame_data.fg_probability is not None
                    else None
                )
                seen_rgb = frame_data.image_rgb.clone().detach()

                # 112, 112, 3
                seen_xyz = co3d_utils.get_rgbd_points(
                    112, 112,
                    frame_data.camera,
                    frame_data.depth_map,
                    mask,
                )

                full_point_cloud = co3d_utils._load_pointcloud(f'{self.args.co3d_path}/{category}/{sequence_name}/pointcloud.ply', max_points=20000)
                full_point_cloud = pad_point_cloud(full_point_cloud, 20000)
                break
            except Exception as e:
                print(category, sequence_name, 'sampling failed', retry, e)

        seen_rgb = seen_rgb.squeeze(0)
        full_rgb = full_point_cloud._features_list[0]

        return (
            (seen_xyz, seen_rgb),
            (full_point_cloud._points_list[0], full_rgb),
            test_frame,
            (category, sequence_name, seen_idx),
        )

    def __len__(self) -> int:
        n_objs = sum([len(cat_dataset.seq_name2idx.keys()) for cat_dataset in self.all_datasets.values()])
        if self.is_train:
            return int(n_objs * self.args.train_epoch_len_multiplier)
        elif self.is_viz:
            return n_objs
        else:
            return int(n_objs * self.args.eval_epoch_len_multiplier)
