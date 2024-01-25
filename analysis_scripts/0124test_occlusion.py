# %%
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.output_data import OutputData, Transforms, TransformMatrices, EulerAngles, LocalTransforms, ScreenPosition, Bounds, ImageSensors, AvatarTransformMatrices, Occlusion
from tdw.librarian import ModelLibrarian

c = Controller()

object_id = c.get_unique_id()
init_obj_position = {"x": 0.0, "y": 2.0, "z": 0.0}
cam_position = {"x": 5.0, "y": 5.0, "z": 0.0}
cam = ThirdPersonCamera(position=cam_position,
                        look_at=init_obj_position)
c.add_ons.append(cam)
# Create the scene and add the object.
c.communicate([{"$type": "set_screen_size", "width": 600, "height": 800},
               {"$type": "simulate_physics", "value": False},
               c.get_add_scene('iceland_beach'),
               c.get_add_object(model_name="arflex_hollywood_sofa",
                                library="models_core.json",
                                position=init_obj_position,
                                object_id=object_id),
                                # {"$type": "scale_object",
                                #  "id": object_id,
                                #  "scale_factor": {"x": 3, "y": 3, "z": 3}},
                                ])

cam.look_at(None)
c.communicate([])


# %%
resp = c.communicate([
    {"$type": "teleport_object", "position": {"x": 0.0, "y": 0.72, "z": 5.9}, "id": object_id},
    {"$type": "rotate_object_to", "rotation": {"x": -0.12, "y": 0.2, "z": 0.17, "w": 0.96}, "id": object_id},
    {"$type": "teleport_avatar_to", "position": {"x": -1.57, "y": 0.68, "z": 0.25}, "avatar_id": cam.avatar_id},
    {"$type": "rotate_sensor_container_to", "rotation": {"x": 0.05, "y": 0.16, "z": 0.003, "w": 0.98}, "avatar_id": cam.avatar_id},
    {"$type": "send_occlusion"},
    # {"$type": "send_occlusion", "object_ids": [object_id, ], "ids": [cam.avatar_id, ], "frequency": "once"}
])

for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    if r_id == "occl":
        occl = Occlusion(resp[i])
        print("Occlusion avatar_id: ", occl.get_avatar_id())
        print("Occluded: ", occl.get_occluded())
        print("Unccluded: ", occl.get_unoccluded())
        print(f"q: {1 - occl.get_occluded() / occl.get_unoccluded():.2f}")

# %%


# %%



