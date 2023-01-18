import argparse
import os
import torch

from omnidata.omnidata_tools.torch.dataloader.component_datasets.hypersim import HypersimDataset

from prepare_hypersim import load_scene_names


def main(args):
    dataset_cache_folder = '../dataset_cache'
    if not os.path.isdir(dataset_cache_folder):
        os.mkdir(dataset_cache_folder)

    options = HypersimDataset.Options(args.omnidata_starter_dataset_path)
    options.load_mesh_textures = True
    options.cache_dir = './tmp'
    options.multiview_sampling_method = None
    dataset = HypersimDataset(options)

    all_meshes = {}
    for scene_name in load_scene_names(hypersim_path=args.hypersim_path, is_train=False):
        print('loading mesh for', scene_name)
        all_meshes[scene_name] = dataset._load_mesh(scene_name)
    torch.save(all_meshes, f'{dataset_cache_folder}/all_hypersim_val_meshes.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MCC', add_help=False)
    parser.add_argument('--hypersim_path', default='', type=str, help='path to hypersim dataset')
    parser.add_argument('--omnidata_starter_dataset_path', default='', type=str, help='path to hypersim dataset')
    args = parser.parse_args()
    main(args)