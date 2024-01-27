# TDW Image Dataset

Generate a dataset of synthetic images using [TDW](https://github.com/threedworld-mit/tdw).

This repo is based on Esther Alter's repo https://github.com/alters-mit/tdw_image_dataset. The main difference is that this repo logs rich ground-truth information about the latent variables during the image generation process. The latent variables include object distance, rotations, and translation related to the camera. Additionally, the sampling of object positions and poses, and test of occluded objects are adjusted.

## Install
```
git clone https://github.com/YudiXie/tdw_image_dataset.git

conda create -n tdw python=3.9
conda activate tdw
conda install pandas
pip install tdw

cd tdw_image_dataset
python generate_dataset.py -n tdw5k
```
