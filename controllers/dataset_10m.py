import argparse
from pathlib import Path
from tdw_image_dataset.image_dataset import ImageDataset

"""
Generate a dataset that have around 10M images, 10M for training, 100K for testing
that have all categories
multiple scenes
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='', help='the directory to save the dataset to')
    args = parser.parse_args()

    dataset_folder = "tdw_image_dataset_10m"

    if args.directory != '':
        output_dir = Path(args.directory).joinpath(dataset_folder)
    else:
        output_dir = Path.home().joinpath(dataset_folder)

    # 10 scenes, 8 outdoors, 2 indoors
    scenes = [
              "box_room_2018",
              "building_site",
              "dead_grotto",
              "downtown_alleys", # has 6 sub regions, okay with 0.5 offset, rendered from within the building, too noisy
              "iceland_beach",
              "lava_field",
              "ruin",
              "savanna_flat_6km",
              "suburb_scene_2023",
              "tdw_room",
              ]
    
    c = ImageDataset(
                     num_img_total=10100000,
                     output_directory=output_dir,
                     materials=False,
                     launch_build=True, # for local machine
                     scene_list=scenes,
                     )

    # Generate a "partial" dataset per scene.
    c.generate_multi_scene()
