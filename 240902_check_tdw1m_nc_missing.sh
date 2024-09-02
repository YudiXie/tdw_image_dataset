#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=normal
#SBATCH -e /om/weka/dicarlo/yu_xie/projects/tdw_image_dataset/slurm_output/slurm-%j-check_missing.out
#SBATCH -o /om/weka/dicarlo/yu_xie/projects/tdw_image_dataset/slurm_output/slurm-%j-check_missing.out

source ~/.bashrc
echo -e "System Info: \n----------\n$(hostnamectl)\n----------"
cd /om/user/yu_xie/projects/tdw_image_dataset
conda activate mtvision3
python check_missing.py --index $1
echo "Check dataset $1 finished!"
