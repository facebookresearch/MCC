# Dataset Preparation

## CO3D v2
1. Please follow [the official instruction](https://github.com/facebookresearch/co3d) to download the dataset.
2. We use the dataset provider in [Implicitron](https://github.com/facebookresearch/pytorch3d/tree/main/pytorch3d/implicitron) for data loading. To speed up the loading, we cache the loaded meta data. Please run 
```
cd scripts
python prepare_co3d.py --co3d_path [path to CO3D data]
```
to generate the cache. The cached data take ~4.1GB of space.

## Hypersim
1. Please follow [the official instruction](https://github.com/apple/ml-hypersim) to download the dataset.
2. We preprocess the Hypersim data for faster loading:
```
cd scripts
python prepare_hypersim.py --hypersim_path [path to Hypersim data]
```
The resulting data take ~19GB of space.

3. We evaluate our method using the ground-truth meshes, which are available for purchase [here](https://www.turbosquid.com/Search/3D-Models?include_artist=evermotion). We borrow the dataloader from [Omnidata](https://github.com/EPFL-VILAB/omnidata) for loading the meshes. To speed up dataloading, we preprocess the data by
```
python prepare_hypersim_mesh.py --omnidata_starter_dataset_path [path to omnidata starter dataset] --hypersim_path [path to Hypersim data]
```
The resulting data take ~5.5GB of space.
