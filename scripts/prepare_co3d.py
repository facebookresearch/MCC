# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os
import sys
sys.path.append('../')
from collections import defaultdict

import torch

from util.co3d_utils import get_dataset_map


def main(args):
    dataset_cache_folder = '../dataset_cache'
    if not os.path.isdir(dataset_cache_folder):
        os.mkdir(dataset_cache_folder)
    all_categories = [c.split('/')[-1] for c in list(glob.glob(args.co3d_path + '/*')) if not c.endswith('.json')]

    for category in all_categories:
        print(f'Loading dataset map ({category})')
        dataset_map = get_dataset_map(
            args.co3d_path,
            category,
            'fewview_dev',
        )

        for split in ['train', 'val']:
            dataset = dataset_map[split]
            seq_name2idx = defaultdict(list)
            for i, ann in enumerate(dataset.frame_annots):
                seq_name2idx[ann["frame_annotation"].sequence_name].append(i)
            dataset.seq_name2idx = seq_name2idx
            torch.save(dataset, f'{dataset_cache_folder}/{category}_{split}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MCC', add_help=False)
    parser.add_argument('--co3d_path', default='', type=str, help='path to CO3D dataset')
    args = parser.parse_args()
    main(args)