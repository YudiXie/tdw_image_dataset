from pathlib import Path
from tdw_image_dataset.image_dataset import ImageDataset

"""
Generate a dataset that is of same size as the HvM dataset
4608 training iamges and 1152 testing images
only have 8 categories
multiple scenes
"""

# TODO: haven't been tested
if __name__ == "__main__":
    output_dir = Path.home().joinpath("tdw_image_dataset_small_multi_env")
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

    scenes = ["building_site",
              "lava_field",
              "iceland_beach",
              "ruin",
              "dead_grotto",
              "abandoned_factory"]
    train = int(4608 / len(scenes))
    val = int(1152 / len(scenes))
    
    c = ImageDataset(train=train,
                     val=val,
                     hdri=False,
                     overwrite=False,
                     output_directory=output_dir,
                     launch_build=True,
                     subset_wnids=subset_ids,
                     do_zip=False,
                     terminate_build=False,
                     )

    # Generate a "partial" dataset per scene.
    for scene, i in zip(scenes, range(len(scenes))):
        print(f"{scene}\t{i + 1}/{len(scenes)}")
        c.run(scene_name=scene)
    # Terminate the build.
    c.communicate({"$type": "terminate"})

    # Zip.
    c.zip_images()
