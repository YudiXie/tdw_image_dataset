import argparse
import sys
from datetime import datetime
from pathlib import Path
from tqdm import trange

import pandas as pd


if __name__ == '__main__':
    """
    Check if all of the images and image meta files specified in the index exist.
    If not, save the missing images to a csv file, and remove the corresponding
    scene and model from processed_records. So that the next time we run
    the image generation scripts, the missing images will be generated.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='', help='the path to saved index')
    # eg. '/om2/user/yu_xie/data/tdw_images/tdw_image_dataset_small_multi_env_hdri/index_img_5898.csv'
    parser.add_argument('-s', '--scene', default='all', help='names of the scene to check')
    args = parser.parse_args()

    index_path = Path(args.index)
    dataset_path = index_path.parent

    full_df = pd.read_csv(index_path, index_col=0)
    print(f'Checking {index_path} ...')
    if args.scene != 'all':
        full_df = full_df[full_df['scene'] == args.scene]
        print(f'Checking scene: {args.scene} ...')
    else:
        print('Checking all scenes ...')
    
    missing_idx = []
    for idx in trange(len(full_df)):
        row = full_df.iloc[idx]
        image_folder = dataset_path.joinpath('images', row['scene'], row['wnid'], row['model'])
        image_path = image_folder.joinpath(f'img_{row.name:010d}.jpg')
        image_meta_path = image_folder.joinpath(f'img_{row.name:010d}_info.csv')

        # Check if the image file and the image meta file exists
        if not image_path.is_file() or not image_meta_path.is_file():
            missing_idx.append(idx)

    # Convert the list of missing images to a DataFrame
    missing_df = full_df.iloc[missing_idx]
    
    if len(missing_df) == 0:
        print("No missing images. Dataset is complete!")
        complete_time = datetime.now()
        complete_path = dataset_path.joinpath(f'dataset_scene_{args.scene}_complete.txt')
        complete_path.write_text(f'Dataset scene {args.scene} is complete, checked: {complete_time.strftime("%Y-%m-%d %H:%M:%S")}', 
                                 encoding="utf-8")
        dataset_path.joinpath(f'scene_{args.scene}_missing.csv').unlink(missing_ok=True)
        sys.exit("exit program.")
    
    print(f"Scene {args.scene} missing {len(missing_df)} images.")
    missing_df.to_csv(dataset_path.joinpath(f'scene_{args.scene}_missing.csv'))

    # if input('Continue to modify processed_records? (yes/no): ') != 'yes':
    #     sys.exit("exit program.")
    
    # # remove scenes from processed_scenes.txt
    # missing_scenes = missing_df['scene'].unique()
    # done_scenes_path = dataset_path.joinpath("processed_scenes.txt")
    # processed_scenes_names = done_scenes_path.read_text(encoding="utf-8").split("\n")

    # for scene_name in missing_scenes:
    #     processed_scenes_names.remove(scene_name)
    
    # done_scenes_path.write_text("\n".join(processed_scenes_names), encoding="utf-8")
    # print(f"Updated processed_scenes. Removed {len(missing_scenes)} scenes.")
    
    # # remove models from scene_processed_records.txt
    # for scene_name in missing_scenes:
    #     scene_df = missing_df[missing_df['scene'] == scene_name]
    #     undo_models = scene_df['model'].unique()

    #     done_models_path = dataset_path.joinpath(f"{scene_name}_processed_records.txt")
    #     processed_model_names = done_models_path.read_text(encoding="utf-8").split("\n")

    #     # remove undo_models from processed_model_names
    #     for model_name in undo_models:
    #         processed_model_names.remove(model_name)

    #     # write back to done_models_path
    #     done_models_path.write_text("\n".join(processed_model_names), encoding="utf-8")
    #     print(f"Updated {scene_name}_processed_records. Removed {len(undo_models)} models.")
