# %%
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.output_data import OutputData, Transforms, TransformMatrices, EulerAngles, LocalTransforms, ScreenPosition, Bounds, ImageSensors, AvatarTransformMatrices, Occlusion
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
resp = c.communicate([{"$type": "set_screen_size", "width": 600, "height": 800},
               {"$type": "simulate_physics", "value": False},
               c.get_add_scene('box_room_4x5'),
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
offset = 0.5
print("Checking scene bounds")
print("number region", len(scene_bounds.regions))
for i, region in enumerate(scene_bounds.regions):
    assert region.x_max - region.x_min > 2 * offset, f"{i} region x too small"
    assert region.z_max - region.z_min > 2 * offset, f"{i} region z too small"
    assert region.y_max > 0.4 + offset, f"{i} region y too small"
print("Scene bounds ok")

# %%

region_number = np.random.randint(0, len(scene_bounds.regions))
print(region_number, len(scene_bounds.regions))
region = scene_bounds.regions[region_number]

x_min = region.x_min + offset
x_max = region.x_max - offset
z_min = region.z_min + offset
z_max = region.z_max - offset
y_max = region.y_max - offset

avatar_p = np.array([np.random.uniform(x_min, x_max),
                     np.random.uniform(0.4, y_max),
                     np.random.uniform(z_min, z_max)])

cam_rot = np.random.randn(4)
cam_rot = cam_rot / np.linalg.norm(cam_rot)
resp = c.communicate([
    {"$type": "teleport_avatar_to", "position": TDWUtils.array_to_vector3(avatar_p), "avatar_id": cam.avatar_id},
    {"$type": "rotate_sensor_container_to",
    "rotation": TDWUtils.array_to_vector4(cam_rot),
    "avatar_id": cam.avatar_id,
    },
])

# %%


# %%



