import argparse
from pathlib import Path
from tdw_image_dataset.image_dataset import ImageDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='', help='name of the dataset')
    parser.add_argument('-s', '--scenes', nargs='+', default=[], help='names of the scenes to generate')
    parser.add_argument('-d', '--directory', default='', help='the directory to save the dataset to')
    args = parser.parse_args()

    if args.name == 'tdw5k':
        """
        Generate a dataset that is of same size as the HvM dataset
        around 4608 training iamges and 1152 testing images
        only have 8 categories
        multiple scenes
        """

        dataset_folder = "tdw_image_dataset_5k"
        num_img_total = 4608 + 1152
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
    
    elif args.name == 'tdw1m':
        """
        Generate a dataset that is of same size Imagenet
        around 1,300,000 training images and 50,000 testing images
        that have all categories
        multiple scenes
        """

        dataset_folder = "tdw_image_dataset_1m"
        num_img_total = 1300000 + 50000
        subset_ids = None
    
    elif args.name == 'tdw10m':
        """
        Generate a dataset that have around 10M images, 10M for training, 100K for testing
        that have all categories
        multiple scenes
        """

        dataset_folder = "tdw_image_dataset_10m"
        num_img_total = 10000000 + 100000
        subset_ids = None
        
    else:
        raise NotImplementedError("Unknown dataset name")
    
    if args.directory != '':
        output_dir = Path(args.directory).joinpath(dataset_folder)
    else:
        output_dir = Path.home().joinpath(dataset_folder)
    
    # 10 scenes, 8 outdoors, 2 indoors
    scenes = [
        "box_room_2018", # indoor
        "building_site",
        "dead_grotto",
        "downtown_alleys",
        "iceland_beach",
        "lava_field",
        "ruin",
        "savanna_flat_6km",
        "suburb_scene_2023",
        "tdw_room", # indoor
        ]
    
    c = ImageDataset(
        num_img_total=num_img_total,
        output_directory=output_dir,
        subset_wnids=subset_ids,
        scene_list=scenes,
        scene_to_generate=args.scenes,
        )

    c.generate_multi_scene()
