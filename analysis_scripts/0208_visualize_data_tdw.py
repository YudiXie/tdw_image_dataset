# %%
from pathlib import Path
import pandas as pd
import numpy as np

from visualizer import show_image_and_meta, load_image_and_meta

from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelLibrarian

# %%
index_path = Path('Y:/tdw_images/tdw_image_dataset_1m_20240206/index_img_1349370.csv')
dset_path = index_path.parent
dset_index = pd.read_csv(index_path, index_col=0)
dset_index['image_index'] = dset_index.index

# %%
lib = ModelLibrarian(library="models_core.json")
c = Controller()

y_offset = 2.0
# set the camrea looking towards the z direction
init_obj_position = {"x": 0.0, "y": y_offset, "z": 2.0}
cam_position = {"x": 0.0, "y": y_offset, "z": 0.0}

# Create the scene
c.communicate([
    TDWUtils.create_empty_room(12, 12),
    {"$type": "create_avatar", "type": "A_Img_Caps_Kinematic", "id": 'a'},
    {"$type": "simulate_physics", "value": False},
    {"$type": "set_img_pass_encoding", "value": False},
    {'$type': 'set_field_of_view', 'field_of_view': 60},
    {'$type': 'set_camera_clipping_planes', 'far': 160, 'near': 0.01},
    {"$type": "set_anti_aliasing", "mode": "subpixel"},
    {"$type": "set_aperture", "aperture": 70},
    {"$type": "set_screen_size", "width": 512, "height": 512},
    {"$type": "teleport_avatar_to", "position": cam_position, "avatar_id": "a"},
    {"$type": "look_at_position", "position": init_obj_position,  "avatar_id": "a"}
])

first_run = True

# %%
if not first_run:
    c.communicate([
        {"$type": "destroy_object", "id": object_id},
        {"$type": "remove_position_markers", "id": marker_id},
    ])

# show the image and meta data
i_ = np.random.randint(len(dset_index))
img_index, img, img_meta = load_image_and_meta(dset_path, dset_index.iloc[i_])
show_image_and_meta(img, img_meta)

# show the rendered image in TDW
model_name = img_meta['record_name']
record = lib.get_record(model_name)
scale = TDWUtils.get_unit_scale(record)

object_id = c.get_unique_id()
marker_id = c.get_unique_id()

object_position = {
    "x": img_meta["rel_pos_x"],
    "y": img_meta["rel_pos_y"] + y_offset,
    "z": img_meta["rel_pos_z"],
}
object_rotation = {
    "x": img_meta["rel_rot_x"],
    "y": img_meta["rel_rot_y"],
    "z": img_meta["rel_rot_z"],
    "w": img_meta["rel_rot_w"],
}

c.communicate([
    c.get_add_object(model_name=model_name,
                     library="models_core.json",
                     position=init_obj_position,
                     object_id=object_id),
    {"$type": "scale_object", "id": object_id, "scale_factor": {"x": scale, "y": scale, "z": scale}},
    {"$type": "rotate_object_to", "rotation": object_rotation, "id": object_id},
    {"$type": "teleport_object", "id": object_id, "position": object_position, "use_centroid": True},
    {"$type": "add_position_marker", "position": object_position, "id": marker_id},
])

first_run = False
