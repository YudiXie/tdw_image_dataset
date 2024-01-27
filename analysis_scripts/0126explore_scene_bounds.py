# %%
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.output_data import OutputData, Transforms, TransformMatrices, EulerAngles, LocalTransforms, ScreenPosition, Bounds, ImageSensors, AvatarTransformMatrices, Occlusion, AvatarKinematic
from tdw.librarian import ModelLibrarian, SceneLibrarian
from tdw.scene_data.scene_bounds import SceneBounds
from tdw.scene_data.region_bounds import RegionBounds
import numpy as np

# %%
librarian = SceneLibrarian()
for record in librarian.records:
    print(record.name, record.location)

# %%
c = Controller()

object_id = c.get_unique_id()
init_obj_position = {"x": 0.0, "y": 2.0, "z": 0.0}
cam_position = {"x": 5.0, "y": 5.0, "z": 0.0}
cam = ThirdPersonCamera(position=cam_position,
                        look_at=init_obj_position)
c.add_ons.append(cam)
# Create the scene and add the object.
resp = c.communicate([{"$type": "set_screen_size", "width": 1000, "height": 800},
               {"$type": "simulate_physics", "value": False},
               c.get_add_scene('downtown_alleys'),
               c.get_add_object(model_name="arflex_hollywood_sofa",
                                library="models_core.json",
                                position=init_obj_position,
                                object_id=object_id),
                {"$type": "send_scene_regions"},
                ])
scene_bounds = SceneBounds(resp=resp)

cam.look_at(None)
c.communicate([])

# %%
c1 = {"r": 1, "g": 0, "b": 0, "a": 1}
c2 = {"r": 0, "g": 1, "b": 0, "a": 1}
c3 = {"r": 0, "g": 0, "b": 1, "a": 1}
c4 = {"r": 1, "g": 1, "b": 0, "a": 1}
c5 = {"r": 1, "g": 0, "b": 1, "a": 1}
c6 = {"r": 0, "g": 1, "b": 1, "a": 1}
colors = [c1, c2, c3, c4, c5, c6]

# %%
# add position markers
import itertools
from collections import defaultdict
positions = defaultdict(list)
for reg_i, region in enumerate(scene_bounds.regions):
    for x, y, z in itertools.product([region.x_min, region.x_max], [0.4, region.y_max], [region.z_min, region.z_max]):
        pos = {"x": x, "y": y, "z": z}
        print(pos, 'color', reg_i)
        positions[reg_i].append(pos)

# %%
for i in range(6):
    for pos in positions[i]:
        marker_id = c.get_unique_id()
        c.communicate([
            {"$type": "add_position_marker", "position": pos, "id": marker_id, "scale": 2.0, "color": colors[i]},
        ])

# %%
c.communicate([
    {"$type": "teleport_avatar_by", "position": {"x": 0, "y": 1, "z": 0}, "avatar_id": cam.avatar_id},
])

# %%
c.communicate([
    {"$type": "rotate_sensor_container_by", "axis": "yaw", "angle": -10, "avatar_id": cam.avatar_id},
])

# %%
resp = c.communicate([
    {"$type": "send_avatars", "ids": [cam.avatar_id, ], "frequency": "once"}
])
for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    if r_id == "avki":
        ak = AvatarKinematic(resp[i])
        assert ak.get_avatar_id() == cam.avatar_id
        cam_pos = ak.get_position()
        print(cam_pos)


