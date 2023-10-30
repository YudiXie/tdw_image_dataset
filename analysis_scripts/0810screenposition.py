# %%
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.output_data import OutputData, Transforms, TransformMatrices, EulerAngles, LocalTransforms, ScreenPosition

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
               TDWUtils.create_empty_room(12, 12),
               c.get_add_object(model_name="arflex_hollywood_sofa",
                                library="models_core.json",
                                position=init_obj_position,
                                object_id=object_id)])

# %%
obj_position = {"x": 0.0, "y": 2.0, "z": 0.0}
resp = c.communicate([{"$type": "teleport_object", "position": obj_position, "id": object_id},
                      {"$type": "send_screen_positions", "position_ids": [0], "positions": [obj_position]}])
for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    if r_id == "scre":
        scene_positions = ScreenPosition(resp[i])
        print(scene_positions.get_screen())

# %%
resp = c.communicate([
    {"$type": "rotate_object_by", "axis": "roll", "angle": -20, "id": object_id, "is_world": True, "use_centroid": True},
    {"$type": "send_transforms", "ids": [object_id], "frequency": "once"},
    {"$type": "send_transform_matrices", "ids": [object_id], "frequency": "once"},
    {"$type": "send_euler_angles", "ids": [object_id], "frequency": "once"},
    ])
for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    if r_id == "tran":
        obj_transforms = Transforms(resp[i])
        print('number of objects: ', obj_transforms.get_num())
        for idx in range(obj_transforms.get_num()):
            print('Obj id: ', obj_transforms.get_id(idx))
            print('Obj Position: ', obj_transforms.get_position(idx).round(3))
            print('Obj Rotation: ', obj_transforms.get_rotation(idx).round(3))
            print('Obj Forward: ', obj_transforms.get_forward(idx).round(3))
    if r_id == "trma":
        obj_transforms_m = TransformMatrices(resp[i])
        print('number of objects: ', obj_transforms_m.get_num())
        for idx in range(obj_transforms_m.get_num()):
            print('Obj id: ', obj_transforms_m.get_id(idx))
            print('Obj Trans Matrix:')
            print(obj_transforms_m.get_matrix(idx).round(3))
    if r_id == "eule":
        obj_euler_angles = EulerAngles(resp[i])
        print('number of objects: ', obj_euler_angles.get_num())
        for idx in range(obj_euler_angles.get_num()):
            print('Obj id: ', obj_euler_angles.get_id(idx))
            print('Obj Rotation Euler Angles: ', obj_euler_angles.get_rotation(idx).round(0))

# %%
resp = c.communicate([
    {"$type": "parent_object_to_avatar", "id": object_id, "avatar_id": cam.avatar_id, "sensor": True},
    ])

# %%
resp = c.communicate([
    {"$type": "object_look_at_position", "position": cam_position, "id": object_id},
    ])

# %%
cam.look_at(None)
c.communicate([])

# %%
# cam.teleport(position=cam_position)
# cam.rotate(rotation={"x": 0, "y": 20, "z": 10})
# cam.teleport(position={"x": 2, "y": 2, "z": 0}, absolute=False)
# c.communicate([])

resp = c.communicate([
    {"$type": "rotate_object_by", "axis": "pitch", "angle": -20, "id": object_id, "is_world": True, "use_centroid": False},
    # {"$type": "parent_object_to_avatar", "id": object_id, "avatar_id": cam.avatar_id, "sensor": True},
    # {"$type": "object_look_at_position", "position": cam_position, "id": object_id},
    {"$type": "send_local_transforms", "ids": [object_id], "frequency": "once"},
    {"$type": "send_transforms", "ids": [object_id], "frequency": "once"},
    {"$type": "send_euler_angles", "ids": [object_id], "frequency": "once"},
    # {"$type": "unparent_object", "id": object_id},
    ])
for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    if r_id == "ltra":
        obj_ltransforms = LocalTransforms(resp[i])
        print('Local Transform:')
        print('number of objects: ', obj_ltransforms.get_num())
        for idx in range(obj_ltransforms.get_num()):
            print('Obj id: ', obj_ltransforms.get_id(idx))
            print('Obj Position: ', obj_ltransforms.get_position(idx).round(3))
            print('Obj Rotation: ', obj_ltransforms.get_rotation(idx).round(3))
            print('Obj Forward: ', obj_ltransforms.get_forward(idx).round(3))
            print('Obj Euler Angles: ', obj_ltransforms.get_euler_angles(idx).round(3))
    if r_id == "tran":
        obj_transforms = Transforms(resp[i])
        print('Global Transform:')
        print('number of objects: ', obj_transforms.get_num())
        for idx in range(obj_transforms.get_num()):
            print('Obj id: ', obj_transforms.get_id(idx))
            print('Obj Position: ', obj_transforms.get_position(idx).round(3))
            print('Obj Rotation: ', obj_transforms.get_rotation(idx).round(3))
            print('Obj Forward: ', obj_transforms.get_forward(idx).round(3))
    if r_id == "eule":
        obj_euler_angles = EulerAngles(resp[i])
        print('number of objects: ', obj_euler_angles.get_num())
        for idx in range(obj_euler_angles.get_num()):
            print('Obj id: ', obj_euler_angles.get_id(idx))
            print('Obj Rotation Euler Angles: ', obj_euler_angles.get_rotation(idx).round(0))

# %%
resp = c.communicate([
    {"$type": "unparent_object", "id": object_id},
    ])

# %%
resp = c.communicate([
    {"$type": "send_local_transforms", "ids": [object_id], "frequency": "once"},
    ])
for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    if r_id == "ltra":
        obj_ltransforms = LocalTransforms(resp[i])
        print('number of objects: ', obj_ltransforms.get_num())
        for idx in range(obj_ltransforms.get_num()):
            print('Obj id: ', obj_ltransforms.get_id(idx))
            print('Obj Position: ', obj_ltransforms.get_position(idx).round(3))
            print('Obj Rotation: ', obj_ltransforms.get_rotation(idx).round(3))
            print('Obj Forward: ', obj_ltransforms.get_forward(idx).round(3))
            print('Obj Euler Angles: ', obj_ltransforms.get_euler_angles(idx).round(3))

# %%
c.communicate({"$type": "terminate"})

# %%



