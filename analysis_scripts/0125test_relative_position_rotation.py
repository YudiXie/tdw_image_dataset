# %%
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.output_data import OutputData, Transforms, TransformMatrices, EulerAngles, LocalTransforms, ScreenPosition, Bounds, ImageSensors, AvatarTransformMatrices, CameraTransforms
from tdw.librarian import ModelLibrarian
import numpy as np
from tdw.quaternion_utils import QuaternionUtils


# %%
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
                                ])



# %%
cam.look_at(None)
c.communicate([
])

# %%
object_postion = np.array([1, 0.72, 5.5])
object_rotation = np.array([-0.12, 0.2, 0.17, 0.96])
cam_position = np.array([-1.57, 0.68, 0.25])
cam_rotation = np.array([0.05, 0.16, 0.003, 0.98])

# %%
object_postion = np.random.uniform(0, 1, 3)
object_rotation = np.random.uniform(-1, 1, 4)
object_rotation /= np.linalg.norm(object_rotation)
cam_position = np.random.uniform(0, 1, 3)
cam_rotation = np.random.uniform(-1, 1, 4)
cam_rotation /= np.linalg.norm(cam_rotation)

# %%
c.communicate([
        {"$type": "teleport_object", "position": TDWUtils.array_to_vector3(object_postion), "id": object_id},
        {"$type": "rotate_object_to", "rotation": TDWUtils.array_to_vector4(object_rotation), "id": object_id},
        {"$type": "teleport_avatar_to", "position": TDWUtils.array_to_vector3(cam_position), "avatar_id": cam.avatar_id},
        {"$type": "rotate_sensor_container_to", "rotation": TDWUtils.array_to_vector4(cam_rotation), "avatar_id": cam.avatar_id},
])

c.communicate([
    # {"$type": "parent_object_to_avatar", "id": object_id, "avatar_id": cam.avatar_id, "sensor": False},
    {"$type": "parent_object_to_avatar", "id": object_id, "avatar_id": cam.avatar_id, "sensor": True},
    ])

resp = c.communicate([
    {"$type": "send_transforms", "frequency": "once"},
    {"$type": "send_screen_positions", "position_ids": [0], "positions": [TDWUtils.array_to_vector3(object_postion) ]},
    {"$type": "send_local_transforms", "ids": [object_id], "frequency": "once"},
    {"$type": "send_camera_transforms", "ids": [object_id], "avatar_id": cam.avatar_id, "frequency": "once"},
])

for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    if r_id == "tran":
        print('-------------Global Transform-------------')
        obj_transforms = Transforms(resp[i])
        print('number of objects: ', obj_transforms.get_num())
        for idx in range(obj_transforms.get_num()):
            print('Obj id: ', obj_transforms.get_id(idx))
            print('Obj Forward: ', obj_transforms.get_forward(idx).round(3))
            print('Obj Position: ', obj_transforms.get_position(idx).round(3))
            print('Obj Rotation: ', obj_transforms.get_rotation(idx).round(3))
    if r_id == "ctra":
        print('-------------CameraTransforms-------------')
        camtrans = CameraTransforms(resp[i])
        assert camtrans.get_num() == 1
        assert camtrans.get_id(0) == object_id
        print("CamTrans pos: ", camtrans.get_position(0).round(3))
        print("CamTrans rot: ", camtrans.get_rotation(0).round(3))
        print("CamTrans rot euler: ", QuaternionUtils.quaternion_to_euler_angles(camtrans.get_rotation(0)).round(3))
    if r_id == "ltra":
        print('-------------Local Transform-------------')
        obj_ltransforms = LocalTransforms(resp[i])
        print('number of objects: ', obj_ltransforms.get_num())
        for idx in range(obj_ltransforms.get_num()):
            print('Obj id: ', obj_ltransforms.get_id(idx))
            print('Obj Forward: ', obj_ltransforms.get_forward(idx).round(3))
            print('Obj Position: ', obj_ltransforms.get_position(idx).round(3))
            print('Obj Rotation: ', obj_ltransforms.get_rotation(idx).round(3))
            print('Obj Euler Angles: ', obj_ltransforms.get_euler_angles(idx).round(3))
    if r_id == "scre":
        print('-------------Screen Position-------------')
        scene_positions = ScreenPosition(resp[i])
        print("Screen Position", np.array(scene_positions.get_screen()).round(3))

print('-------------Calculated-------------')
rel_pos = QuaternionUtils.world_to_local_vector(object_postion, cam_position, cam_rotation)
rel_rot = QuaternionUtils.multiply(QuaternionUtils.get_inverse(cam_rotation), object_rotation)
rel_rot_euler = QuaternionUtils.quaternion_to_euler_angles(rel_rot).round(3)
print('Relative Position: ', rel_pos.round(3))
print('Relative Rotation: ', rel_rot.round(3))
print('Relative Rotation Euler: ', rel_rot_euler.round(3))

c.communicate([
    {"$type": "unparent_object", "id": object_id},
    ])


# %%



