import argparse
import sys
from pathlib import Path
from tqdm import trange

import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='', help='the path to saved index')
    # eg. '/om2/user/yu_xie/data/tdw_images/tdw_image_dataset_small_multi_env_hdri/index_img_5898.csv'
    args = parser.parse_args()

    index_path = Path(args.index)
    dataset_path = index_path.parent

    full_df = pd.read_csv(index_path, index_col=0)

    missing_idx = []
    for idx in trange(len(full_df)):
        row = full_df.iloc[idx]
        image_folder = dataset_path.joinpath('images', row['scene'], row['wnid'], row['model'])
        image_path = image_folder.joinpath(f'img_img_{idx:010d}.jpg')
        image_meta_path = image_folder.joinpath(f'img_{idx:010d}_info.csv')

        # Check if the image file and the image meta file exists
        if not image_path.is_file() or not image_meta_path.is_file():
            missing_idx.append(idx)

    # Convert the list of missing images to a DataFrame
    missing_df = full_df.iloc[missing_idx]
    
    if len(missing_df) == 0:
        print("No missing images.")
        sys.exit("exit program.")
    
    print(f"Missing {len(missing_df)} images.")
    missing_df.to_csv(dataset_path.joinpath('missing.csv'))

    if input('Continue to modify processed_records? (yes/no): ') != 'yes':
        sys.exit("exit program.")
    
    # remove scenes from processed_scenes.txt
    missing_scenes = missing_df['scene'].unique()
    done_scenes_path = dataset_path.joinpath("processed_scenes.txt")
    processed_scenes_names = done_scenes_path.read_text(encoding="utf-8").split("\n")

    for scene_name in missing_scenes:
        processed_scenes_names.remove(scene_name)
    
    done_scenes_path.write_text("\n".join(processed_scenes_names), encoding="utf-8")
    print(f"Updated processed_scenes. Removed {len(missing_scenes)} scenes.")
    
    # remove models from scene_processed_records.txt
    for scene_name in missing_scenes:
        scene_df = missing_df[missing_df['scene'] == scene_name]
        undo_models = scene_df['model'].unique()

        done_models_path = dataset_path.joinpath(f"{scene_name}_processed_records.txt")
        processed_model_names = done_models_path.read_text(encoding="utf-8").split("\n")

        # remove undo_models from processed_model_names
        for model_name in undo_models:
            processed_model_names.remove(model_name)

        # write back to done_models_path
        done_models_path.write_text("\n".join(processed_model_names), encoding="utf-8")
        print(f"Updated {scene_name}_processed_records. Removed {len(undo_models)} models.")
