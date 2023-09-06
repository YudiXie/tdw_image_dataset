from pathlib import Path
from tdw_image_dataset.image_dataset import ImageDataset

"""
Generate a dataset that is of same size Imagenet
around 1,300,000 training images and 50,000 testing images
that have all categories
multiple scenes
"""


if __name__ == "__main__":
    output_dir = Path.home().joinpath("tdw_image_dataset_large")

    scenes = ["suburb_scene_2018",
              "building_site",
              "lava_field",
              "iceland_beach",
              "ruin",
              "dead_grotto",
              ]
    
    train = int(4608 / len(scenes))
    val = int(1152 / len(scenes))
    
    c = ImageDataset(train=train,
                     val=val,
                     output_directory=output_dir,
                     launch_build=True, # for local machine
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
