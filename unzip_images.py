import shutil
import argparse
from pathlib import Path
from datetime import datetime



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', nargs='+', default=[], help='names of the scene to unpack')
    parser.add_argument('-d', '--directory', default='', help='the directory of the dataset')
    args = parser.parse_args()

    extract_names = args.name
    if extract_names:
        images_path = Path(args.directory).joinpath('images')
        for name in extract_names:
            print(f"Unzipping {name}... Start time: {datetime.now()}")
            shutil.unpack_archive(images_path.joinpath(f"{name}.zip"), 
                                  images_path.joinpath(f"{name}"))
            print(f"Unzipping {name}...done. End time: {datetime.now()}")
    else:
        print("No scene name provided, nothing to unzip")
    