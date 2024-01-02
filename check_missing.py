import os
import argparse
import sys
from pathlib import Path

import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='', help='the directory to saved dataset')
    args = parser.parse_args()
    
    dataset_folder = args.directory
    missing_index = 1
    # Path to the index file
    index_file = os.path.join(dataset_folder, 'index.csv')

    # Read the index file into a DataFrame
    ds_index = pd.read_csv(index_file)

    # List to store information about missing images
    missing_images = []

    # Iterate over each row in the DataFrame
    for index, row in ds_index.iterrows():
        # Construct the expected file path for the image
        image_folder = os.path.join(dataset_folder, 'images', row['scene'], row['wnid'], row['model'])
        image_path = os.path.join(image_folder, f'img_{row["index"]:010d}.png')
        image_meta_path = os.path.join(image_folder, f'img_{row["index"]:010d}_info.csv')

        # Check if the image file exists
        if not os.path.exists(image_path) or not os.path.exists(image_meta_path):
            # Record the missing image information
            missing_images.append(row)

    # Convert the list of missing images to a DataFrame
    missing_df = pd.DataFrame(missing_images)
    
    print(f"Missing {len(missing_df)} images.")

    if input('Continue to modify processed_records? (yes/no): ') != 'yes':
        sys.exit("exit program.")
    
    unique_scenes = missing_df['scene'].unique()
    for scene_name in unique_scenes:
        scene_df = missing_df[missing_df['scene'] == scene_name]
        undo_models = scene_df['model'].unique()

        done_models_path: Path = Path(dataset_folder).joinpath(f"{scene_name}_processed_records.txt")
        processed_model_names = done_models_path.read_text(encoding="utf-8").split("\n")

        # remove undo_models from processed_model_names
        processed_model_names = list(set(processed_model_names) - set(undo_models))

        # write back to done_models_path
        done_models_path.write_text("\n".join(processed_model_names), encoding="utf-8")
        print(f"Updated {scene_name}_processed_records.txt")
        