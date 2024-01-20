# %%
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.output_data import OutputData, Transforms, TransformMatrices, EulerAngles, LocalTransforms, ScreenPosition, Bounds
from tdw.librarian import ModelLibrarian

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
    {"$type": "rotate_object_by", "axis": "roll", "angle": -20, "id": object_id},
])


# %%
resp = c.communicate([
    {"$type": "rotate_object_by", "axis": "pitch", "angle": -20, "id": object_id},
])

# %%
resp = c.communicate([{"$type": "send_bounds", "ids": [object_id, ]}])
for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    if r_id == "boun":
        b = Bounds(resp[i])
        print(b.get_id(0))
        

# %%
place_marker(c, b.get_bottom(0))

# %%



