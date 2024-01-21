# %%
from pathlib import Path
import pandas as pd
import numpy as np

from visualizer import show_image_and_meta

# %%
index_path = Path('/om2/user/yu_xie/data/tdw_images//tdw_image_dataset_small_multi_env_hdri/index_img_5898.csv')
dset_path = index_path.parent
dset_index = pd.read_csv(index_path, index_col=0)

# %%
dset_index['image_index'] = dset_index.index

# %%
dset_index

# %%
i_ = np.random.randint(len(dset_index))
show_image_and_meta(dset_path, dset_index.iloc[i_])
