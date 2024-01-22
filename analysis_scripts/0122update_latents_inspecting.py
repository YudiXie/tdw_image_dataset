# %%
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.output_data import OutputData, Transforms, TransformMatrices, EulerAngles, LocalTransforms, ScreenPosition, Bounds, ImageSensors, AvatarTransformMatrices, Occlusion
from tdw.librarian import ModelLibrarian

import numpy as np
RNG = np.random.RandomState(1111)

# %%
model_name = 'arflex_hollywood_sofa'

lib = ModelLibrarian(library="models_core.json")
record = None
for r in lib.records:
    if r.name == model_name:
        record = r

# %%
c = Controller()

object_id = c.get_unique_id()
init_obj_position = {"x": 0.0, "y": 2.0, "z": 0.0}
cam_position = {"x": 1.0, "y": 2.0, "z": 0.0}
cam = ThirdPersonCamera(position=cam_position,
                        look_at=init_obj_position)
c.add_ons.append(cam)
# Create the scene and add the object.
scale = TDWUtils.get_unit_scale(record)
c.communicate([{"$type": "set_screen_size", "width": 256, "height": 256},
               {"$type": "simulate_physics", "value": False},
               TDWUtils.create_empty_room(12, 12),
            #    c.get_add_scene('iceland_beach'),
               c.get_add_object(model_name=model_name,
                                library="models_core.json",
                                position=init_obj_position,
                                object_id=object_id),
                                {"$type": "scale_object",
                                 "id": object_id,
                                 "scale_factor": {"x": scale, "y": scale, "z": scale}},
                                ])

# %%
cam.look_at(None)
c.communicate([])

# %%
rot_range = 45
cam_rot_range = 30
c.communicate([ 
                {"$type": "teleport_object", "position": init_obj_position, "id": object_id},
                {
                "$type": "object_look_at_position",
                "position": cam_position, 
                "id": object_id,
                },
                {
                "$type": "rotate_object_by",
                "id": object_id,
                "angle": RNG.uniform(-rot_range, rot_range),
                "axis": "pitch",
                # "use_centroid": True,  # will change object pivot position
                "is_world": False,
                },
                                {
                "$type": "rotate_object_by",
                "id": object_id,
                "angle": RNG.uniform(-rot_range, rot_range),
                "axis": "yaw",
                # "use_centroid": True,  # will change object pivot position
                "is_world": False,
                },
                                {
                "$type": "rotate_object_by",
                "id": object_id,
                "angle": RNG.uniform(-rot_range, rot_range),
                "axis": "roll",
                # "use_centroid": True,  # will change object pivot position
                "is_world": False,
                },
                            {"$type": "look_at",
             "object_id": object_id,
             "use_centroid": True,
             "avatar_id": cam.avatar_id,
             },
                            {"$type": "rotate_sensor_container_by",
             "angle": RNG.uniform(-cam_rot_range, cam_rot_range),
             "axis": "pitch",
             "avatar_id": cam.avatar_id,
             },
            {"$type": "rotate_sensor_container_by",
             "angle": RNG.uniform(-cam_rot_range, cam_rot_range),
             "axis": "yaw",
             "avatar_id": cam.avatar_id,
             },
                ])

# %%

def place_coords(controller):
    place_marker(controller, 0, 0, 0)
    place_marker(controller, 1, 0, 0)
    place_marker(controller, 2, 0, 0)
    place_marker(controller, 0, 1, 0)
    place_marker(controller, 0, 2, 0)
    place_marker(controller, 0, 0, 1)
    place_marker(controller, 0, 0, 2)


def place_marker(controller, position):
    x, y, z = position.tolist()
    marker_id = controller.get_unique_id()
    scale = 3.0
    controller.communicate([controller.get_add_object(model_name="aaa_battery",
                                                      library="models_core.json",
                                                      position={"x": x, "y": y, "z": z},
                                                      object_id=marker_id),
                            {"$type": "scale_object",
                             "id": marker_id,
                             "scale_factor": {"x": scale, "y": scale, "z": scale}},
                             {"$type": "set_color",
                              "color": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}, 
                              "id": marker_id},
                             ])

# %%
resp = c.communicate([

    {"$type": "teleport_object", "position": {"x": 0.0, "y": 0.72, "z": 5.5}, "id": object_id},
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
        print("Occlusion sensor name: ", occl.get_sensor_name())
        print("Occlusion value: ", occl.get_occluded())


# %%
resp = c.communicate([
                        {
                "$type": "object_look_at_position",
                "position": cam_position, 
                "id": object_id,
                },
    # {"$type": "teleport_object", "position": {"x": 0.0, "y": 2.0, "z": 0.0}, "id": object_id},
    # {"$type": "teleport_object_by", "position": {"x": 0.0, "y": 1.0, "z": 0.0}, "id": object_id},
    # {"$type": "rotate_object_to", "rotation": {"x": -0.12, "y": 0.2, "z": 0.17, "w": 0.96}, "id": object_id},
    # {"$type": "teleport_avatar_to", "position": {"x": 5.0, "y": 5.0, "z": 0.0}, "avatar_id": cam.avatar_id},
    # {"$type": "rotate_sensor_container_to", "rotation": {"x": 0.05, "y": 0.16, "z": 0.003, "w": 0.98}, "avatar_id": cam.avatar_id},
    # {"$type": "send_transforms", "ids": [object_id], "frequency": "once"},
                #     {
                # "$type": "rotate_object_by",
                # "id": object_id,
                # "angle": 10,
                # "axis": "pitch",
                # "is_world": False,
                # },
    {"$type": "send_transforms", "frequency": "once"},
    {"$type": "send_bounds", "ids": [object_id, ]},
    {"$type": "send_occlusion"},
])

for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    if r_id == "occl":
        occl = Occlusion(resp[i]).get_occluded()
        print("Occlusion: ", occl)
    if r_id == "tran":
        obj_transforms = Transforms(resp[i])
        print('number of objects: ', obj_transforms.get_num())
        for idx in range(obj_transforms.get_num()):
            print('Obj id: ', obj_transforms.get_id(idx))
            print('Obj Position: ', obj_transforms.get_position(idx).round(3))
            print('Obj Rotation: ', obj_transforms.get_rotation(idx).round(3))
            print('Obj Forward: ', obj_transforms.get_forward(idx).round(3))

for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    if r_id == "boun":
        b = Bounds(resp[i])
print("get_bottom: ", b.get_bottom(0).round(3))
print("get_center: ", b.get_center(0).round(3))

# %%
resp = c.communicate([
    {"$type": "teleport_object", "position": {"x": 0.0, "y": 0.72, "z": 5.5}, "id": object_id},
    {"$type": "rotate_object_to", "rotation": {"x": -0.12, "y": 0.2, "z": 0.17, "w": 0.96}, "id": object_id},
    {"$type": "teleport_avatar_to", "position": {"x": -1.57, "y": 0.68, "z": 0.25}, "avatar_id": cam.avatar_id},
    {"$type": "rotate_sensor_container_to", "rotation": {"x": 0.05, "y": 0.16, "z": 0.003, "w": 0.98}, "avatar_id": cam.avatar_id},
    {"$type": "send_transforms", "ids": [object_id], "frequency": "once"},
    {"$type": "send_bounds", "ids": [object_id, ]},
    {"$type": "send_occlusion", "object_ids": [object_id, ], "ids": [cam.avatar_id, ], "frequency": "once"}
])

for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    if r_id == "occl":
        occl = Occlusion(resp[i])
        print("Occlusion avatar_id: ", occl.get_avatar_id())
        print("Occlusion sensor name: ", occl.get_sensor_name())
        print("Occlusion value: ", occl.get_occluded())
    if r_id == "tran":
        obj_transforms = Transforms(resp[i])
        print('number of objects: ', obj_transforms.get_num())
        for idx in range(obj_transforms.get_num()):
            print('Obj id: ', obj_transforms.get_id(idx))
            print('Obj Position: ', obj_transforms.get_position(idx).round(3))
            print('Obj Rotation: ', obj_transforms.get_rotation(idx).round(3))
            print('Obj Forward: ', obj_transforms.get_forward(idx).round(3))

for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    if r_id == "boun":
        b = Bounds(resp[i])
print(b.get_bottom(0).round(3))

# %%
resp = c.communicate([
    {"$type": "rotate_sensor_container_to", "rotation": {"w": 0.6, "x": 3.5, "y": -45, "z": 0}, "avatar_id": cam.avatar_id},
                #    {"$type": "teleport_avatar_to",
                # "avatar_id": cam.avatar_id,
                # "position": {"x": 3, "y": 4, "z": 5.5}},
    {"$type": "send_image_sensors", "ids": [cam.avatar_id, ], "frequency": "once"},
    {"$type": "send_avatar_transform_matrices"},
])

for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    if r_id == "imse":
        img_sensors = ImageSensors(resp[i])
        print('avator id, ', img_sensors.get_avatar_id())
        print('number of sensor: ', img_sensors.get_num_sensors())
        for idx in range(img_sensors.get_num_sensors()):
            print('sensory name: ', img_sensors.get_sensor_name(idx))
            print('sensor on: ', img_sensors.get_sensor_on(idx))
            print('sensor rotation: ', img_sensors.get_sensor_rotation(idx))
            print('sensor forward: ', img_sensors.get_sensor_forward(idx))
            print('sensor FOV: ', img_sensors.get_sensor_field_of_view(idx))
    
    if r_id == "atrm":
        atrm = AvatarTransformMatrices(resp[i])
        print('number of avators: ', atrm.get_num())
        for idx in range(atrm.get_num()):
            print('avators id: ', atrm.get_id(idx))
            print('get_avatar_matrix: ', atrm.get_avatar_matrix(idx))
            print('get_sensor_matrix: ', atrm.get_sensor_matrix(idx))


# %%
resp = c.communicate([
    # {"$type": "rotate_sensor_container_by", "axis": "roll", "angle": 10, "avatar_id": cam.avatar_id},
    {"$type": "rotate_sensor_container_to", "rotation": {"w": -0.6630, "x": -0.13473, "y": 0.7260, "z": -0.1227}, "avatar_id": cam.avatar_id},
            #         {"$type": "look_at",
            #  "object_id": object_id,
            #  "use_centroid": True,
            #  "avatar_id": cam.avatar_id},
    {"$type": "send_image_sensors", "ids": [cam.avatar_id, ], "frequency": "once"},
    {"$type": "send_avatar_transform_matrices"},
])

for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    if r_id == "imse":
        img_sensors = ImageSensors(resp[i])
        print('avator id, ', img_sensors.get_avatar_id())
        print('number of sensor: ', img_sensors.get_num_sensors())
        for idx in range(img_sensors.get_num_sensors()):
            print('sensory name: ', img_sensors.get_sensor_name(idx))
            print('sensor on: ', img_sensors.get_sensor_on(idx))
            print('sensor rotation: ', img_sensors.get_sensor_rotation(idx))
            print('sensor forward: ', img_sensors.get_sensor_forward(idx))
            print('sensor FOV: ', img_sensors.get_sensor_field_of_view(idx))
    
    if r_id == "atrm":
        atrm = AvatarTransformMatrices(resp[i])
        print('number of avators: ', atrm.get_num())
        for idx in range(atrm.get_num()):
            print('avators id: ', atrm.get_id(idx))
            print('get_avatar_matrix: ', atrm.get_avatar_matrix(idx))
            print('get_sensor_matrix: ', atrm.get_sensor_matrix(idx))


# %%
c.communicate({"$type": "terminate"})

# %%



