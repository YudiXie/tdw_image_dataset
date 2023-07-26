from pathlib import Path
from tdw_image_dataset.image_dataset import ImageDataset

"""
Generate a dataset that is of same size as the HvM dataset
4608 training iamges and 1152 testing images
"""

if __name__ == "__main__":
    c = ImageDataset(train=4608, val=1152, 
                     library="models_core.json", 
                     materials=False, 
                     hdri=False,
                     do_zip=True,
                     random_seed=0,
                     output_directory=Path.home().joinpath("tdw_image_dataset_small"),
                     launch_build=True)
    c.run(scene_name="building_site")
