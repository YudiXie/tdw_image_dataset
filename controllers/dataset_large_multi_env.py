import argparse
from pathlib import Path
from tdw_image_dataset.image_dataset import ImageDataset

"""
Generate a dataset that is of same size Imagenet
around 1,300,000 training images and 50,000 testing images
that have all categories
multiple scenes
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='', help='the directory to save the dataset to')
    args = parser.parse_args()

    if args.directory != '':
        output_dir = Path(args.directory)
    else:
        output_dir = Path.home().joinpath("tdw_image_dataset_large")

    scenes = [
              "building_site",
              "lava_field",
              "iceland_beach",
              "ruin",
              "dead_grotto",
              "suburb_scene_2018",
              ]
    
    c = ImageDataset(
                     num_img_total=(1300000 + 50000),
                     output_directory=output_dir,
                     materials=False,
                     launch_build=True, # for local machine
                     scene_list=scenes,
                     )

    # Generate a "partial" dataset per scene.
    c.generate_multi_scene()
