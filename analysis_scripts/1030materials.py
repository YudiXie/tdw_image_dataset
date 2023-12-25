# %%
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.librarian import ModelLibrarian
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH


# %%
lib = ModelLibrarian(library="models_core.json")
records = lib.get_all_models_in_wnid('n02774152')

# %%
for r in records:
    print(r.name)
    print(r.substructure)

# %%
c = Controller()
object_id = c.get_unique_id()

model_record = ModelLibrarian().get_record("b04_armani_handbag")
cam = ThirdPersonCamera(position={"x": 0.5, "y": 1.4, "z": -0.15},
                        look_at=object_id)
path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("set_visual_material")
print(f"Images will be saved to: {path}")
cap = ImageCapture(avatar_ids=[cam.avatar_id], pass_masks=["_img"], path=path)
c.add_ons.extend([cam, cap])

c.communicate([
               TDWUtils.create_empty_room(12, 12),
               c.get_add_object(model_name=model_record.name,
                                object_id=object_id),
               ])


# %%
cam.teleport(position={"x": -0.0, "y": -0.1, "z": 0.0}, absolute=False)
c.communicate([])

# %%
material_name = "rubber_black"
c.communicate([
               c.get_add_material(material_name=material_name,
                                  library="materials_low.json"),
               {"$type": "set_visual_material",
                "material_index": 0,
                "material_name": material_name,
                "object_name": "Box014",
                "id": object_id},
                ])

# %%
c.communicate({"$type": "terminate"})

# %%



