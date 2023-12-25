# %%
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, Images
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH


# %%

"""
Create a scene, add an object, and save the image.
"""
c = Controller()
object_id = c.get_unique_id()

commands = [TDWUtils.create_empty_room(12, 12),
            c.get_add_object(model_name="arflex_hollywood_sofa",
                             position={"x": 0, "y": 0, "z": 0},
                             object_id=object_id)]
commands.extend([{"$type": "create_avatar", "type": "A_Img_Caps_Kinematic", "id": "a"},
                 {"$type": "teleport_avatar_to", "avatar_id": "a", "position": {"x": 10, "y": 8, "z": -3}},
                 {"$type": "look_at_position", "position": {"x": 0, "y": 0.0, "z": 0}},
                 ]
                )
commands.extend([{"$type": "set_pass_masks",
                  "pass_masks": ["_img"],
                  "avatar_id": "a"},
                 {"$type": "send_images",
                  "frequency": "always",
                  "ids": ["a"]}])
commands.extend([c.get_add_hdri_skybox(skybox_name="autumn_hockey_4k"),
                 {"$type": "rotate_hdri_skybox_by", "angle": 30},
                 ])

resp = c.communicate(commands)
output_directory = str(EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("send_images").resolve())
print(f"Images will be saved to: {output_directory}")

for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    # Get Images output data.
    if r_id == "imag":
        print(f"Got images! {i}")
        images = Images(resp[i])
        # Determine which avatar captured the image.
        if images.get_avatar_id() == "a":
            # Iterate throught each capture pass.
            for j in range(images.get_num_passes()):
                # This is the _img pass.
                if images.get_pass_mask(j) == "_img":
                    image_arr = images.get_image(j)
                    # Get a PIL image.
                    pil_image = TDWUtils.get_pil_image(images=images, index=j)
            # Save the image.
            TDWUtils.save_images(images=images, filename="0", output_directory=output_directory)


# %%
resp =c.communicate([
               {"$type": "rotate_hdri_skybox_by", "angle": 30},
               ])

# %%
for i in range(len(resp) - 1):
    r_id = OutputData.get_data_type_id(resp[i])
    # Get Images output data.
    if r_id == "imag":
        print(f"Got images! {i}")
        images = Images(resp[i])
        # Determine which avatar captured the image.
        if images.get_avatar_id() == "a":
            # Iterate throught each capture pass.
            for j in range(images.get_num_passes()):
                # This is the _img pass.
                if images.get_pass_mask(j) == "_img":
                    image_arr = images.get_image(j)
                    # Get a PIL image.
                    pil_image = TDWUtils.get_pil_image(images=images, index=j)
            # Save the image.
            TDWUtils.save_images(images=images, filename="0", output_directory=output_directory)

# %%
pil_image

# %%
pil_image # original image

# %%
pil_image # image rotated 30 degrees

# %%
c.communicate({"$type": "terminate"})

# %%
c = Controller()
camera = ThirdPersonCamera(avatar_id="a",
                           position={"x": -4.28, "y": 0.85, "z": 4.27},
                           look_at={"x": 0, "y": 0, "z": 0})
c.add_ons.append(camera)
c.communicate([
               c.get_add_scene(scene_name="lava_field"),
               ])


# %%
c.imgs_per_skybox

# %%
c.communicate([
               c.get_add_hdri_skybox(skybox_name="cave_wall_4k"),
               ])

# %%
c.communicate({"$type": "rotate_hdri_skybox_by", "angle": 10, "axis": "yaw"})

# %%
camera.look_at(None)
c.communicate([])

# %%
camera.teleport(position={"x": 0, "y": 1, "z": 0}, absolute=False)
c.communicate([])

# %%
camera.rotate(rotation={"x": 0, "y": 0, "z": 10})
c.communicate([])

# %%
c.communicate({"$type": "terminate"})

# %%
from tdw.librarian import SceneLibrarian

librarian = SceneLibrarian()
for record in librarian.records:
    print(record.name, record.hdri)

# %%
from tdw.librarian import HDRISkyboxLibrarian

librarian = HDRISkyboxLibrarian()
for record in librarian.records:
    print(record.name, record.location)

# %%
list = [s for s in librarian.records if s.location == "exterior"]

# %%
for record in list:
    print(record.name, record.location)

# %%



