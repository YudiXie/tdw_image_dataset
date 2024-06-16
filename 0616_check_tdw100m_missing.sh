#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=normal
#SBATCH -e /om/weka/dicarlo/yu_xie/projects/tdw_image_dataset/slurm_output/slurm-%j-240616_check_tdw100m_missing.out
#SBATCH -o /om/weka/dicarlo/yu_xie/projects/tdw_image_dataset/slurm_output/slurm-%j-240616_check_tdw100m_missing.out

source ~/.bashrc
echo -e "System Info: \n----------\n$(hostnamectl)\n----------"
cd /om/user/yu_xie/projects/tdw_image_dataset
conda activate mtvision
python check_missing.py --index /om/user/yu_xie/data/tdw_images/tdw_image_dataset_100m_20240222/index_img_100100440.csv -s $1
echo "Check scene $1 finished!"
