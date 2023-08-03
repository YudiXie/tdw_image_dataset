from pathlib import Path
from tdw_image_dataset.image_dataset import ImageDataset

"""
Generate a dataset that is of same size as the HvM dataset
4608 training iamges and 1152 testing images
"""

if __name__ == "__main__":
    subset_ids = [
        'n02774152', # 'bag, handbag, pocketbook, purse’, 12 records
        'n02933112', # 'cabinet’, 33 records
        'n03001627', # 'chair’, 25 records
        'n03761084', # 'microwave oven’, 12 records
        'n03880531', # 'pan’, 12 records
        'n04256520', # ‘sofa’, 14 records
        'n04379243', # ‘table’, 20 records
        'n04461879', # ‘toy’, 12 records
    ]
    c = ImageDataset(train=4608, val=1152,
                     hdri=False,
                     output_directory=Path.home().joinpath("tdw_image_dataset_small"),
                     launch_build=True,
                     subset_wnids=subset_ids,
                     do_zip=True,
                     )
    c.run(scene_name="building_site")
