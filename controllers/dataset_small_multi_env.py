from pathlib import Path
from tdw_image_dataset.image_dataset import ImageDataset

"""
Generate a dataset that is of same size as the HvM dataset
around 4608 training iamges and 1152 testing images
only have 8 categories
multiple scenes
"""


if __name__ == "__main__":
    output_dir = Path.home().joinpath("tdw_image_dataset_small_multi_env_hdri")
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

    scenes = ["suburb_scene_2018",
              "building_site",
              "lava_field",
              "iceland_beach",
              "ruin",
              "dead_grotto",
              ]
    
    c = ImageDataset(
                     num_img_total=(4608 + 1152),
                     output_directory=output_dir,
                     materials=True,
                     launch_build=True, # for local machine
                     subset_wnids=subset_ids, # only 8 categories
                     do_zip=False,
                     terminate_build=False,
                     scene_list=scenes,
                     )

    # Generate a "partial" dataset per scene.
    c.run_multi_scene()

    # Terminate the build.
    c.communicate({"$type": "terminate"})

    # Zip.
    c.zip_images()
