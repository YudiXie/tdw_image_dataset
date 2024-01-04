import os
import argparse
import sys
from pathlib import Path
from tqdm import trange

import pandas as pd


def find_missing_df(dataset_folder):
    index_file = os.path.join(dataset_folder, 'index_img_5898.csv')
    full_df = pd.read_csv(index_file, index_col=0)

    missing_idx = []
    for idx in trange(len(full_df)):
        row = full_df.iloc[idx]
        image_folder = os.path.join(dataset_folder, 'images', row['scene'], row['wnid'], row['model'])
        image_path = os.path.join(image_folder, f'img_img_{idx:010d}.jpg')
        image_meta_path = os.path.join(image_folder, f'img_{idx:010d}_info.csv')

        # Check if the image file and the image meta file exists
        if not os.path.exists(image_path) or not os.path.exists(image_meta_path):
            missing_idx.append(idx)

    # Convert the list of missing images to a DataFrame
    return full_df.iloc[missing_idx]


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--directory', default='', help='the directory to saved dataset')
    # # eg. '/om2/user/yu_xie/data/tdw_images/tdw_image_dataset_small_multi_env_hdri'
    # args = parser.parse_args()
    # dataset_folder = args.directory

    dataset_folder = '/om2/user/yu_xie/data/tdw_images/tdw_image_dataset_small_multi_env_hdri'
    missing_df = find_missing_df(dataset_folder)
    
    if len(missing_df) == 0:
        print("No missing images.")
        sys.exit("exit program.")
    
    print(f"Missing {len(missing_df)} images.")
    missing_df.to_csv(os.path.join(dataset_folder, 'missing.csv'))

    if input('Continue to modify processed_records? (yes/no): ') != 'yes':
        sys.exit("exit program.")
    
    for scene_name in missing_df['scene'].unique():
        scene_df = missing_df[missing_df['scene'] == scene_name]
        undo_models = scene_df['model'].unique()

        done_models_path: Path = Path(dataset_folder).joinpath(f"{scene_name}_processed_records.txt")
        processed_model_names = done_models_path.read_text(encoding="utf-8").split("\n")

        # remove undo_models from processed_model_names
        for model_name in undo_models:
            processed_model_names.remove(model_name)

        # write back to done_models_path
        done_models_path.write_text("\n".join(processed_model_names), encoding="utf-8")
        print(f"Updated {scene_name}_processed_records. Removed {len(undo_models)} models.")
