#!/bin/bash
for i in /om/user/yu_xie/data/tdw_images/tdw_image_dataset_100m_20240222/images/*.zip
do
  sbatch 0616_check_tdw100m_missing.sh "`basename $1i .zip`"
done
