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
                                ])



# %%
cam.look_at(None)
c.communicate([
        # {"$type": "teleport_object", "position": {"x": 0.0, "y": 0.72, "z": 5.5}, "id": object_id},
        {"$type": "rotate_object_to", "rotation": {"x": -0.3, "y": 0.4, "z": 0.6, "w": 0.5}, "id": object_id},
        {"$type": "teleport_avatar_to", "position": {"x": -1.57, "y": 0.68, "z": 0.25}, "avatar_id": cam.avatar_id},
        {"$type": "rotate_sensor_container_to", "rotation": {"x": 0.24, "y": 0.1, "z": 0.003, "w": 0.98}, "avatar_id": cam.avatar_id},
])

resp = c.communicate([
    {"$type": "teleport_object", "position": {"x": -2.0, "y": 5.0, "z": 4.5}, "id": object_id, "use_centroid": True},
    {"$type": "send_transforms", "frequency": "once"},
    {"$type": "send_bounds", "ids": [object_id, ]},
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
    if r_id == "boun":
        b = Bounds(resp[i])
        assert b.get_num() == 1
        assert b.get_id(0) == object_id
        print("bounds.get_bottom: ", b.get_bottom(0).round(3))
        print("bounds.get_center: ", b.get_center(0).round(3))



