import sys
import subprocess
from pathlib import Path

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

shtext = [
    '#!/bin/bash',
    '#SBATCH -t 2:00:00',
    '#SBATCH -N 1',
    '#SBATCH -n 4',
    '#SBATCH --mem=32G',
    '#SBATCH --partition=dicarlo',
    '#SBATCH -e /om/user/yu_xie/projects/tdw_image_dataset/slurm_output/slurm-%j.out',
    '#SBATCH -o /om/user/yu_xie/projects/tdw_image_dataset/slurm_output/slurm-%j.out',
    'source ~/.bashrc',
    'conda activate tdw',
    'cd /om/user/yu_xie/projects/tdw_image_dataset',
    'echo -e "System Info: \\n----------\\n$(hostnamectl)\\n----------"',
    ]


if __name__ == '__main__':
    shscript_folder = Path('/om/user/yu_xie/projects/tdw_image_dataset/shscripts')
    for i_s, scene in enumerate(scenes):
        shscript_path = shscript_folder.joinpath(f'unzip_{scene}.sh')
        cmd = f'python unzip_images.py -d /om/user/yu_xie/data/tdw_images/tdw_image_dataset_10m -n {scene}'
        write_text = shtext + [cmd, ]
        shscript_path.write_text('\n'.join(write_text), encoding='utf-8')
    
    if input('Continue to submit batch jobs? (yes/no): ') != 'yes':
        sys.exit("exit program.")
    
    for scene in scenes:
        shscript_path = shscript_folder.joinpath(f'unzip_{scene}.sh')
        cp_process = subprocess.run(['sbatch', str(shscript_path)],
                                    capture_output=True, check=True)
        cp_stdout = cp_process.stdout.decode()
        print(cp_stdout)
    