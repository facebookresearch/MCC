# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
from omegaconf import DictConfig
from typing import Optional

import torch

from pytorch3d.implicitron.dataset.dataset_map_provider import DatasetMap
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    JsonIndexDatasetMapProviderV2
)
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.io import IO
from pytorch3d.renderer import (
    NDCMultinomialRaysampler,
    ray_bundle_to_ray_points,
)
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds


HOLDOUT_CATEGORIES = set([
    'apple',
    'baseballglove',
    'cup',
    'ball',
    'toyplane',
    'handbag',
    'book',
    'carrot',
    'suitcase',
    'bowl',
])

def get_dataset_map(
    dataset_root: str,
    category: str,
    subset_name: str,
) -> DatasetMap:
    """
    Obtain the dataset map that contains the train/val/test dataset objects.
    """
    expand_args_fields(JsonIndexDatasetMapProviderV2)
    dataset_map_provider = JsonIndexDatasetMapProviderV2(
        category=category,
        subset_name=subset_name,
        dataset_root=dataset_root,
        test_on_train=False,
        only_test_set=False,
        load_eval_batches=True,
        dataset_JsonIndexDataset_args=DictConfig({"remove_empty_masks": False, "load_point_clouds": False}),
    )
    return dataset_map_provider.get_dataset_map()


def _load_pointcloud(pcl_path, max_points):
    pcl = IO().load_pointcloud(pcl_path)
    if max_points > 0:
        pcl = pcl.subsample(max_points)

    return pcl


def get_all_dataset_maps(co3d_path, holdout_categories):
    all_categories = [c.split('/')[-1] for c in list(glob.glob(co3d_path + '/*')) if not c.endswith('.json')]
    all_categories = sorted(all_categories, key=lambda x: hash(x))

    # Obtain the CO3Dv2 dataset map
    train_dataset_maps = {}
    val_dataset_maps = {}
    for category in all_categories:

        print(f'Loading dataset map ({category})')
        dataset_map = {
            'train': torch.load(f'dataset_cache/{category}_train.pt'),
            'val': torch.load(f'dataset_cache/{category}_val.pt')
        }
        if not holdout_categories or category not in HOLDOUT_CATEGORIES:
            train_dataset_maps[category] = dataset_map['train']
        if not holdout_categories or category in HOLDOUT_CATEGORIES:
            val_dataset_maps[category] = dataset_map['val']

    print('Loaded', len(train_dataset_maps), 'categores for train')
    print('Loaded', len(val_dataset_maps), 'categores for val')
    return train_dataset_maps, val_dataset_maps


def get_rgbd_points(
    imh, imw,
    camera: CamerasBase,
    depth_map: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    mask_thr: float = 0.5,
) -> Pointclouds:
    """
    Given a batch of images, depths, masks and cameras, generate a colored
    point cloud by unprojecting depth maps to the  and coloring with the source
    pixel colors.
    """
    depth_map = torch.nn.functional.interpolate(
        depth_map,
        size=[imh, imw],
        mode="bilinear",
        align_corners=False,
    )
    # convert the depth maps to point clouds using the grid ray sampler
    pts_3d = ray_bundle_to_ray_points(
        NDCMultinomialRaysampler(
            image_width=imw,
            image_height=imh,
            n_pts_per_ray=1,
            min_depth=1.0,
            max_depth=1.0,
        )(camera)._replace(lengths=depth_map[:, 0, ..., None])
    ).squeeze(3)[None]

    pts_mask = depth_map > 0.0
    if mask is not None:
        mask = torch.nn.functional.interpolate(
            mask,
            size=[imh, imw],
            mode="bilinear",
            align_corners=False,
        )
        pts_mask *= mask > mask_thr
    pts_3d[~pts_mask] = float('inf')
    return pts_3d.squeeze(0).squeeze(0)

