# TDW Image Dataset

Generate a dataset of synthetic images using [TDW](https://github.com/threedworld-mit/tdw).

This repo is based on Esther Alter's repo https://github.com/alters-mit/tdw_image_dataset. The main difference is that this repo logs rich ground-truth information about the latent variables during the image generation process. The latent variables include object distance, rotations, and translation related to the camera. Additionally, the sampling of object positions and poses, and test of occluded objects are adjusted.

## Install
```bash
git clone https://github.com/YudiXie/tdw_image_dataset.git

conda create -n tdw python=3.9
conda activate tdw
conda install pandas tqdm
pip install tdw

cd tdw_image_dataset
python generate_dataset.py -n tdw5k
```

## Usage
Check mission images in a dataset:

```bash
python check_missing.py --index /om/user/yu_xie/data/tdw_images/tdw_image_dataset_1m/index_img_1349370.csv
```

Unzip images
```bash
python unzip_images.py -d /om/user/yu_xie/data/tdw_images/tdw_image_dataset_1m -n box_room_2018 building_site dead_grotto downtown_alleys iceland_beach lava_field ruin savanna_flat_6km suburb_scene_2023 tdw_room
```
