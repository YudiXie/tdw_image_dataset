import os
from secrets import token_urlsafe
from pathlib import Path
import json
from datetime import datetime
from threading import Thread
from time import time
from typing import List, Dict, Tuple, Optional, Union
from zipfile import ZipFile
from distutils import dir_util
import numpy as np
import pandas as pd
from tqdm import tqdm
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, Occlusion, Images, ImageSensors, Transforms, Version, ScreenPosition, LocalTransforms
from tdw.librarian import ModelLibrarian, MaterialLibrarian, HDRISkyboxLibrarian, ModelRecord, HDRISkyboxRecord
from tdw.scene_data.scene_bounds import SceneBounds
from tdw.scene_data.region_bounds import RegionBounds
from tdw.release.pypi import PyPi
from tdw_image_dataset.image_position import ImagePosition

# The required version of TDW.
REQUIRED_TDW_VERSION: str = "1.9.0"
RNG: np.random.RandomState = np.random.RandomState(0)


class ImageDataset(Controller):
    """
    Generate image datasets. Each image will have a single object in the scene in a random position and orientation.
    Optionally, the scene might have variable lighting and the object might have variable visual materials.

    The image dataset includes all models in a model library (the default library is models_core.json) sorted by wnid and model name.
    """

    """:class_var
    The ID of the avatar.
    """
    AVATAR_ID: str = "a"

    def __init__(self,
                 output_directory: Union[str, Path],
                 port: int = 1071,
                 launch_build: bool = False,
                 materials: bool = False,
                 screen_width: int = 256,
                 screen_height: int = 256,
                 output_scale: float = 1,
                 hdri: bool = True,
                 show_objects: bool = True,
                 clamp_rotation: bool = True,
                 max_height: float = 0.5,
                 occlusion: float = 0.45,
                 less_dark: bool = True,
                 id_pass: bool = False,
                 do_zip: bool = True,
                 train: int = 1300000,
                 val: int = 50000,
                 library: str = "models_core.json",
                 random_seed: int = 0,
                 subset_wnids: Optional[List[str]] = None,
                 offset: float = 0.0,
                 terminate_build: bool = True,
                 ):
        """
        :param output_directory: The path to the root output directory.
        :param port: The port used to connect to the build.
        :param launch_build: If True, automatically launch the build. Always set this to False on a Linux server.
        :param materials: If True, set random visual materials for each sub-mesh of each object.
        :param screen_width: The screen width of the build in pixels.
        :param screen_height: The screen height of the build in pixels.
        :param output_scale: Scale the images by this factor before saving to disk.
        :param hdri: If True, use a random HDRI skybox per frame.
        :param show_objects: If True, show objects.
        :param clamp_rotation: If true, clamp the rotation to +/- 30 degrees around each axis.
        :param max_height: The percentage of the environment height that is the ceiling for the avatar and object. Must be between 0 and 1.
        :param occlusion: The occlusion threshold. Lower value = slower FPS, better composition. Must be between 0 and 1.
        :param less_dark: If True, there will be more daylight exterior skyboxes (requires hdri == True)
        :param id_pass: If True, send and save the _id pass.
        :param do_zip: If True, zip the directory at the end.
        :param train: The number of train images.
        :param val: The number of val images.
        :param library: The path to the library records file.
        :param random_seed: The random seed.
        :param subset_wnids: create a subset only use these wnid categories.
        :param offset: Restrict the agent from offset to the edge of the region.
        :param terminate_build: If True, terminate the build when one scene is done.
        """

        global RNG
        RNG = np.random.RandomState(random_seed)

        if isinstance(output_directory, str):
            """:field
            The root output directory.
            """
            self.output_directory: Path = Path(output_directory)
        else:
            self.output_directory: Path = output_directory
        if not self.output_directory.exists():
            self.output_directory.mkdir(parents=True)
        """:field
        The images output directory.
        """
        self.images_directory: Path = self.output_directory.joinpath("images")
        if not self.images_directory.exists():
            self.images_directory.mkdir(parents=True)
        """:field
        The path to the metadata file.
        """
        self.metadata_path: Path = self.output_directory.joinpath("metadata.txt")
        """:field
        The path to the image metadata file.
        """
        self.images_meta_directory: Path = self.output_directory.joinpath("images_meta")
        if not self.images_meta_directory.exists():
            self.images_meta_directory.mkdir(parents=True)

        """:field
        The width of the build screen in pixels.
        """
        self.screen_width: int = screen_width
        """:field
        The height of the screen in pixels.
        """
        self.screen_height: int = screen_height
        """:field
        Scale all images to this size in pixels before writing to disk.
        """
        self.output_size: Tuple[int, int] = (int(screen_width * output_scale), int(screen_height * output_scale))
        """:field
        If True, scale images before writing to disk.
        """
        self.scale: bool = output_scale != 1
        """:field
        If True, show objects.
        """
        self.show_objects: bool = show_objects
        """:field
        If true, clamp the rotation to +/- 30 degrees around each axis.
        """
        self.clamp_rotation: bool = clamp_rotation
        """:field
        The percentage of the environment height that is the ceiling for the avatar and object. Must be between 0 and 1.
        """
        self.max_height: float = max_height
        """:field
        The occlusion threshold. Lower value = slower FPS, better composition. Must be between 0 and 1.
        """
        self.occlusion: float = occlusion
        """:field
        If True, send and save the _id pass.
        """
        self.id_pass: bool = id_pass
        """:field
        If True, zip the directory at the end.
        """
        self.do_zip: bool = do_zip
        """:field
        The number of train images.
        """
        self.train: int = train
        """:field
        The number of val images.
        """
        self.val: int = val
        """:field
        Restrict the agent from offset to the edge of the region.
        """
        self.offset = offset
        """:field
        If True, terminate the build when one scene is done.
        """
        self.terminate_build = terminate_build

        self.subset_wnids = subset_wnids
        self.current_scene = ''

        assert 0 < max_height <= 1.0, f"Invalid max height: {max_height}"
        assert 0 < occlusion <= 1.0, f"Invalid occlusion threshold: {occlusion}"

        """:field
        If True, there will be more daylight exterior skyboxes (requires hdri == True)
        """
        self.less_dark: bool = less_dark
        """:field
        Cached model substructure data.
        """
        self.substructures: Dict[str, List[dict]] = dict()
        """:field
        Cached initial (canonical) rotations per model.
        """
        self.initial_rotations: Dict[str, Dict[str, float]] = dict()
        """:field
        If True, set random visual materials for each sub-mesh of each object.
        """
        self.materials: bool = materials
        super().__init__(port=port, launch_build=launch_build, check_version=False)
        resp = self.communicate({"$type": "send_version"})
        for i in range(len(resp) - 1):
            if OutputData.get_data_type_id(resp[i]) == "vers":
                build_version = Version(resp[i]).get_tdw_version()
                PyPi.required_tdw_version_is_installed(build_version=build_version,
                                                       required_version=REQUIRED_TDW_VERSION,
                                                       comparison=">=")
        """:field
        The name of the model library file.
        """
        self.model_library_file: str = library
        # Cache the libraries.
        Controller.MODEL_LIBRARIANS[self.model_library_file] = ModelLibrarian(library=self.model_library_file)
        Controller.MATERIAL_LIBRARIANS["materials_low.json"] = MaterialLibrarian("materials_low.json")
        Controller.HDRI_SKYBOX_LIBRARIANS["hdri_skyboxes.json"] = HDRISkyboxLibrarian()
        """:field
        Cached skybox records.
        """
        self.skyboxes: Optional[List[HDRISkyboxRecord]] = None
        # Get skybox records.
        if hdri:
            self.skyboxes: List[HDRISkyboxRecord] = Controller.HDRI_SKYBOX_LIBRARIANS["hdri_skyboxes.json"].records
            # Prefer exterior daytime skyboxes by adding them multiple times to the list.
            if self.less_dark:
                temp = self.skyboxes[:]
                for skybox in temp:
                    if skybox.location != "interior" and skybox.sun_elevation >= 145:
                        self.skyboxes.append(skybox)

    def initialize_scene(self, scene_name) -> SceneBounds:
        """
        Initialize the scene.

        :param scene_name: str, The name of the scene.

        :return: The [`SceneBounds`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/scene_bounds.md) of the scene.
        """

        # Initialize the scene.
        self.current_scene = scene_name
        # Add the avatar.
        commands = [self.get_add_scene(scene_name),
                    {"$type": "create_avatar",
                     "type": "A_Img_Caps_Kinematic",
                     "id": ImageDataset.AVATAR_ID}]
        # Disable physics.
        # Enable jpgs.
        # Set FOV.
        # Set clipping planes.
        # Set AA.
        # Set aperture.
        # Disable vignette.
        commands.extend([{"$type": "simulate_physics",
                          "value": False},
                         {"$type": "set_img_pass_encoding",
                          "value": False},
                         {'$type': 'set_field_of_view',
                          'field_of_view': 60},
                         {'$type': 'set_camera_clipping_planes',
                          'far': 160,
                          'near': 0.01},
                         {"$type": "set_anti_aliasing",
                          "mode": "subpixel"},
                         {"$type": "set_aperture",
                          "aperture": 70},
                         {"$type": "send_scene_regions"}])

        # If we're using HDRI skyboxes, send additional favorable post-process commands.
        if self.skyboxes is not None:
            commands.extend([{"$type": "set_post_exposure",
                              "post_exposure": 0.6},
                             {"$type": "set_contrast",
                              "contrast": -20},
                             {"$type": "set_saturation",
                              "saturation": 10},
                             {"$type": "set_screen_space_reflections",
                              "enabled": False},
                             {"$type": "set_shadow_strength",
                              "strength": 1.0}])
        # Send the commands.
        resp = self.communicate(commands)
        return SceneBounds(resp)

    def generate_metadata(self, scene_name: str) -> None:
        """
        Generate a metadata file for this dataset.

        :param scene_name: The scene name.
        """

        data = {"dataset": str(self.output_directory.resolve()),
                "scene": scene_name,
                "train": self.train,
                "val": self.val,
                "materials": self.materials is not None,
                "hdri": self.skyboxes is not None,
                "screen_width": self.screen_width,
                "screen_height": self.screen_height,
                "output_scale": self.scale,
                "clamp_rotation": self.clamp_rotation,
                "show_objects": self.show_objects,
                "max_height": self.max_height,
                "occlusion": self.occlusion,
                "less_dark": self.less_dark,
                "start": datetime.now().strftime("%H:%M %d.%m.%y")}
        self.metadata_path.write_text(json.dumps(data, sort_keys=True, indent=4))

    def run(self, scene_name: str) -> None:
        """
        Generate the dataset.

        :param scene_name: The scene name.
        """

        # Create the metadata file.
        self.generate_metadata(scene_name=scene_name)

        # Initialize the scene.
        scene_bounds: SceneBounds = self.initialize_scene(scene_name)

        if self.subset_wnids is None:
            # Fetch the WordNet IDs.
            wnids = Controller.MODEL_LIBRARIANS[self.model_library_file].get_model_wnids()
            # Remove any wnids that don't have valid models.
            wnids = [w for w in wnids if len(
                [r for r in Controller.MODEL_LIBRARIANS[self.model_library_file].get_all_models_in_wnid(w)
                if not r.do_not_use]) > 0]
        else:
            wnids = self.subset_wnids
            for w in wnids:
                # check if any models is usable
                assert len([r for r in 
                            Controller.MODEL_LIBRARIANS[self.model_library_file].get_all_models_in_wnid(w) 
                            if not r.do_not_use]) > 0, f"ID: {w} do not have usable models"

        # Set the number of train and val images per wnid.
        num_train = self.train / len(wnids)
        num_val = self.val / len(wnids)

        # Create the progress bar.
        pbar = tqdm(total=len(wnids))

        done_models_path: Path = self.output_directory.joinpath(f"{scene_name}_processed_records.txt")
        # Get a list of models that have already been processed.
        processed_model_names: List[str] = []
        if done_models_path.exists():
            processed_model_names = done_models_path.read_text(encoding="utf-8").split("\n")

        # Iterate through each wnid.
        for w, q in zip(wnids, range(len(wnids))):
            # Update the progress bar.
            pbar.set_description(w)

            # Get all valid models in the wnid.
            records = Controller.MODEL_LIBRARIANS[self.model_library_file].get_all_models_in_wnid(w)
            records = [r for r in records if not r.do_not_use]

            # Remove models that have multiple objects
            records = [r for r in records if r.name not in ['b02_bag', 'lantern_2010', 'b04_bottle_max']]

            # Get the train and val counts.
            train_count = [len(a) for a in np.array_split(
                np.arange(num_train), len(records))][0]
            val_count = [len(a) for a in np.array_split(
                np.arange(num_val), len(records))][0]

            # Process each record.
            fps = "nan"
            for record, i in zip(records, range(len(records))):
                # Set the progress bar description to the wnid and FPS.
                pbar.set_description(f"record {i + 1}/{len(records)}, FPS {fps}")

                # Skip models that have already been processed.
                if record.name in processed_model_names:
                    continue

                # Create all of the images for this model.
                dt = self.process_model(record, scene_bounds, train_count, val_count, w)
                fps = round((train_count + val_count) / dt)

                # Mark this record as processed.
                with done_models_path.open("at") as f:
                    f.write(f"\n{record.name}")
            pbar.update(1)
        pbar.close()

        # aggregate the image meta files
        df_csv_concat = pd.concat(
            [pd.read_csv(str(p), index_col=0) for p in self.images_meta_directory.iterdir()], 
            ignore_index=True)
        df_csv_concat.to_csv(str(self.output_directory.joinpath('images_meta.csv').resolve()))

        # Add the end time to the metadata file.
        metadata = json.loads(self.metadata_path.read_text())
        end_time = datetime.now().strftime("%H:%M %d.%m.%y")
        if "end" in metadata:
            metadata["end"] = end_time
        else:
            metadata.update({"end": end_time})
        self.metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=4))

        # Terminate the build.
        if self.terminate_build:
            self.communicate({"$type": "terminate"})
        # Zip up the images.
        if self.do_zip:
            self.zip_images()

    def _set_skybox(self, records: List[HDRISkyboxRecord], its_per_skybox: int, hdri_index: int, skybox_count: int) -> Tuple[int, int, Optional[dict]]:
        """
        If it's time, set a new skybox.

        :param records: All HDRI records.
        :param its_per_skybox: Iterations per skybox.
        :param hdri_index: The index in the records list.
        :param skybox_count: The number of images of this model with this skybox.

        :return: Data for setting the skybox.
        """

        # Set a new skybox.
        if skybox_count == 0:
            command = self.get_add_hdri_skybox(records[hdri_index].name)
        # It's not time yet to set a new skybox. Don't send a command.
        else:
            command = None
        skybox_count += 1
        if skybox_count >= its_per_skybox:
            hdri_index += 1
            if hdri_index >= len(records):
                hdri_index = 0
            skybox_count = 0
        return hdri_index, skybox_count, command

    def process_model(self, record: ModelRecord, scene_bounds: SceneBounds, train_count: int, val_count: int, wnid: str) -> float:
        """
        Capture images of a model.

        :param record: The model record.
        :param scene_bounds: The bounds of the scene.
        :param train_count: Number of train images for this model in one scene.
        :param val_count: Number of val images for this model in one scene.
        :param wnid: The wnid of the record.
        :return The time elapsed.
        """

        # the index of images generated for this model in this scene
        image_count = 0

        image_positions: List[ImagePosition] = []
        o_id = self.get_unique_id()

        # Add the object.
        resp = self.communicate(self.get_object_initialization_commands(record=record, o_id=o_id))
        # Cache the initial rotation of the object.
        if record.name not in self.initial_rotations:
            self.initial_rotations[record.name] = TDWUtils.array_to_vector4(Transforms(resp[0]).get_rotation(0))
        # The index in the HDRI records array.
        hdri_index = 0
        # The number of iterations on this skybox so far.
        skybox_count = 0
        if self.skyboxes:
            # The number of iterations per skybox for this model.
            its_per_skybox = round((train_count + val_count) / len(self.skyboxes))

            # Set the first skybox.
            hdri_index, skybox_count, skybox_command = self._set_skybox(self.skyboxes, its_per_skybox, hdri_index, skybox_count)
            self.communicate(skybox_command)
        else:
            its_per_skybox = 0

        while len(image_positions) < train_count + val_count:
            # Get a random "room".
            room: RegionBounds = scene_bounds.regions[RNG.randint(0, len(scene_bounds.regions))]
            # Get the occlusion.
            occlusion, image_position = self.get_occlusion(record.name, o_id, room)
            if occlusion < self.occlusion:
                image_positions.append(image_position)
        # Send images.
        # Set the screen size.
        # Set render quality to maximum.
        commands = [{"$type": "send_images",
                     "frequency": "always"},
                    {"$type": "set_pass_masks",
                     "pass_masks": ["_img", "_id"] if self.id_pass else ["_img"]},
                    {"$type": "set_screen_size",
                     "height": self.screen_height,
                     "width": self.screen_width},
                    {"$type": "set_render_quality",
                     "render_quality": 5}]
        # Hide the object maybe.
        if not self.show_objects:
            commands.append({"$type": "hide_object",
                             "id": o_id})
        self.communicate(commands)

        # Generate images from the cached spatial data.
        t0 = time()

        # store image meta data
        image_file_name_list = []

        ty_list = []
        tz_list = []
        neg_x_list = []

        euler_1_list = []
        euler_2_list = []
        euler_3_list = []

        avatar_pos_x_list = []
        avatar_pos_y_list = []
        avatar_pos_z_list = []

        camera_rot_x_list = []
        camera_rot_y_list = []
        camera_rot_z_list = []
        camera_rot_w_list = []
        
        object_pos_x_list = []
        object_pos_y_list = []
        object_pos_z_list = []

        object_rot_x_list = []
        object_rot_y_list = []
        object_rot_z_list = []
        object_rot_w_list = []

        for p in image_positions:
            # Teleport the avatar.
            # Rotate the avatar's camera.
            # Teleport the object.
            # Rotate the object.
            # Get the response.
            commands = [{"$type": "teleport_avatar_to",
                         "position": p.avatar_position},
                        {"$type": "rotate_sensor_container_to",
                         "rotation": p.camera_rotation},
                        {"$type": "teleport_object",
                         "id": o_id,
                         "position": p.object_position},
                        {"$type": "rotate_object_to",
                         "id": o_id,
                         "rotation": p.object_rotation}]
            # Set the visual materials.
            if self.materials is not None:
                if record.name not in self.substructures:
                    self.substructures[record.name] = record.substructure
                for sub_object in self.substructures[record.name]:
                    for i in range(len(sub_object["materials"])):
                        material_name = Controller.MATERIAL_LIBRARIANS["materials_low.json"].records[
                            RNG.randint(0, len(Controller.MATERIAL_LIBRARIANS["materials_low.json"].records))].name
                        commands.extend([self.get_add_material(material_name,
                                                               library="materials_low.json"),
                                         {"$type": "set_visual_material",
                                          "id": o_id,
                                          "material_name": material_name,
                                          "object_name": sub_object["name"],
                                          "material_index": i}])
            # Maybe set a new skybox.
            # Rotate the skybox.
            if self.skyboxes:
                hdri_index, skybox_count, command = self._set_skybox(self.skyboxes, its_per_skybox, hdri_index, skybox_count)
                if command:
                    commands.append(command)
                commands.append({"$type": "rotate_hdri_skybox_by",
                                 "angle": RNG.uniform(0, 360)})

            resp = self.communicate(commands)

            # Create a thread to save the image.
            t = Thread(target=self.save_image, args=(resp, record, self.current_scene, image_count, wnid, train_count))
            t.daemon = True
            t.start()
            image_count += 1

            # instruct the build to send screen position of the object
            # the position is likely the bottom center of the object
            # parent the object to the avatar, and send rotation of the object relative to the camera
            resp = self.communicate(
                [{"$type": "send_screen_positions",
                  "position_ids": [0],
                  "positions": [p.object_position]},
                 {"$type": "parent_object_to_avatar", 
                  "id": o_id, "avatar_id": ImageDataset.AVATAR_ID, 
                  "sensor": True},
                 {"$type": "send_local_transforms", 
                  "ids": [o_id], "frequency": "once"},
                  ])
            
            # unparent the object, to ensure object don't move when the avatar moves
            self.communicate([{"$type": "unparent_object", "id": o_id},])

            has_scre, has_ltra = False, False
            # get the screen position of the object
            for i in range(len(resp) - 1):
                r_id = OutputData.get_data_type_id(resp[i])
                if r_id == "scre":
                    has_scre = True
                    scene_positions = ScreenPosition(resp[i])
                    ty, tz, neg_x = scene_positions.get_screen()
                    ty -= self.screen_width / 2
                    tz -= self.screen_height / 2
                if r_id == "ltra":
                    has_ltra = True
                    obj_ltransforms = LocalTransforms(resp[i])
                    assert obj_ltransforms.get_id(0) == o_id, "object id mismatch"
                    # get the rotation of the object in the screen space
                    relative_euler = obj_ltransforms.get_euler_angles(0)
            assert has_scre and has_ltra, "missing screen position or local transform"

            image_file_name_list.append(f'img_{record.name}_{self.current_scene}_{(image_count - 1):04d}.jpg')

            ty_list.append(ty)  # up-down position, center of image is 0, unit in pixels
            tz_list.append(tz)  # left-right position, center of image is 0, unit in pixels
            neg_x_list.append(neg_x)  # depth of object, unit in 3D space in TDW

            euler_1_list.append(relative_euler[0])
            euler_2_list.append(relative_euler[1])
            euler_3_list.append(relative_euler[2])

            avatar_pos_x_list.append(p.avatar_position['x'])
            avatar_pos_y_list.append(p.avatar_position['y'])
            avatar_pos_z_list.append(p.avatar_position['z'])

            camera_rot_x_list.append(p.camera_rotation['x'])
            camera_rot_y_list.append(p.camera_rotation['y'])
            camera_rot_z_list.append(p.camera_rotation['z'])
            camera_rot_w_list.append(p.camera_rotation['w'])
            
            object_pos_x_list.append(p.object_position['x'])
            object_pos_y_list.append(p.object_position['y'])
            object_pos_z_list.append(p.object_position['z'])

            object_rot_x_list.append(p.object_rotation['x'])
            object_rot_y_list.append(p.object_rotation['y'])
            object_rot_z_list.append(p.object_rotation['z'])
            object_rot_w_list.append(p.object_rotation['w'])
            
        t1 = time()

        # save the meta_data
        save_df = pd.DataFrame.from_dict(
            {
                'scene_name': self.current_scene,
                'wnid': wnid,
                'record_wcategory': record.wcategory,
                'record_name': record.name,
                'image_filename': image_file_name_list,
                'ty': ty_list,
                'tz': tz_list,
                'neg_x': neg_x_list,
                'euler_1': euler_1_list,
                'euler_2': euler_2_list,
                'euler_3': euler_3_list,
                'avatar_pos_x': avatar_pos_x_list,
                'avatar_pos_y': avatar_pos_y_list,
                'avatar_pos_z': avatar_pos_z_list,
                'camera_rot_x': camera_rot_x_list,
                'camera_rot_y': camera_rot_y_list,
                'camera_rot_z': camera_rot_z_list,
                'camera_rot_w': camera_rot_w_list,
                'object_pos_x': object_pos_x_list,
                'object_pos_y': object_pos_y_list,
                'object_pos_z': object_pos_z_list,
                'object_rot_x': object_rot_x_list,
                'object_rot_y': object_rot_y_list,
                'object_rot_z': object_rot_z_list,
                'object_rot_w': object_rot_w_list,
            }
        )

        csv_path = self.images_meta_directory.joinpath(f'{wnid}_{record.name}_{self.current_scene}_meta_data.csv')
        save_df.to_csv(str(csv_path.resolve()))

        # Stop sending images.
        # Destroy the object.
        # Unload asset bundles.
        self.communicate([{"$type": "send_images",
                           "frequency": "never"},
                          {"$type": "destroy_object",
                           "id": o_id},
                          {"$type": "unload_asset_bundles"}])
        return t1 - t0

    def get_object_initialization_commands(self, record: ModelRecord, o_id: int) -> List[dict]:
        """
        :param record: The model record.
        :param o_id: The object ID.

        :return: Commands for creating and initializing the object.
        """

        s = TDWUtils.get_unit_scale(record)
        # Add the object.
        # Set the screen size to 32x32 (to make the build run faster; we only need the average grayscale values).
        # Toggle off pass masks.
        # Set render quality to minimal.
        # Scale the object to "unit size".
        return [{"$type": "add_object",
                 "name": record.name,
                 "url": record.get_url(),
                 "scale_factor": record.scale_factor,
                 "category": record.wcategory,
                 "rotation": record.canonical_rotation,
                 "id": o_id},
                {"$type": "set_screen_size",
                 "height": 32,
                 "width": 32},
                {"$type": "set_pass_masks",
                 "pass_masks": []},
                {"$type": "set_render_quality",
                 "render_quality": 0},
                {"$type": "scale_object",
                 "id": o_id,
                 "scale_factor": {"x": s, "y": s, "z": s}},
                {"$type": "send_transforms"}]

    def save_image(self, resp, record: ModelRecord, scene_name: str, image_count: int, wnid: str, train_count: int) -> None:
        """
        Save an image.

        :param resp: The raw response data.
        :param record: The model record.
        :param image_count: The image count.
        :param wnid: The wnid.
        :param train_count: Total number of train images to generate for this scence and this record.
        """

        # Get the directory.
        directory: Path = self.images_directory.joinpath("train" if image_count < train_count else "val").joinpath(wnid)
        if directory.exists():
            # Try to make the directories. Due to threading, they might already be made.
            try:
                directory.mkdir(parents=True)
            except OSError:
                pass

        # Save the image.
        filename = f"{record.name}_{scene_name}_{image_count:04d}"

        # Save the image without resizing.
        if not self.scale:
            TDWUtils.save_images(Images(resp[0]), filename,
                                 output_directory=directory)
        # Resize the image and save it.
        else:
            TDWUtils.save_images(Images(resp[0]), filename,
                                 output_directory=directory,
                                 resize_to=self.output_size)

    def get_occlusion(self, o_name: str, o_id: int, region: RegionBounds) -> Tuple[float, ImagePosition]:
        """
        Get the "real" grayscale value of an image we hope to capture.

        :param o_name: The name of the object.
        :param o_id: The ID of the object.
        :param region: The scene region bounds.

        :return: (grayscale, distance, avatar_position, object_position, object_rotation, avatar_rotation)
        """

        # Get a random position for the avatar.
        a_p = self.get_avatar_position(region=region, offset=self.offset)
        # Teleport the object.
        commands = self.get_object_position_commands(o_id=o_id, avatar_position=a_p, region=region)
        # Convert the avatar's position to a Vector3.
        a_p = TDWUtils.array_to_vector3(a_p)
        # Teleport the avatar.
        commands.append({"$type": "teleport_avatar_to",
                         "position": a_p})
        # Rotate the object.
        commands.extend(self.get_object_rotation_commands(o_id=o_id, o_name=o_name))
        # Rotate the camera.
        commands.extend(self.get_camera_rotation_commands(o_id=o_id))
        # Request output data.
        commands.extend([{"$type": "send_occlusion",
                          "frequency": "once"},
                         {"$type": "send_image_sensors",
                          "frequency": "once"},
                         {"$type": "send_transforms",
                          "frequency": "once"}])
        # Send the commands.
        resp = self.communicate(commands)

        # Parse the output data:
        # 1. The occlusion value of the image.
        # 2. The camera rotation.
        # 3. The object position and rotation.
        occlusion: float = 0
        cam_rot = None
        o_rot = None
        o_p = None
        for i in range(len(resp) - 1):
            r_id = resp[i][4:8]
            if r_id == b"occl":
                occlusion = Occlusion(resp[i]).get_occluded()
            elif r_id == b"imse":
                cam_rot = ImageSensors(resp[i]).get_sensor_rotation(0)
                cam_rot = {"x": cam_rot[0], "y": cam_rot[1], "z": cam_rot[2], "w": cam_rot[3]}
            elif r_id == b"tran":
                transforms = Transforms(resp[i])
                o_rot = TDWUtils.array_to_vector4(transforms.get_rotation(0))
                o_p = TDWUtils.array_to_vector3(transforms.get_position(0))
        return occlusion, ImagePosition(avatar_position=a_p,
                                        object_position=o_p,
                                        object_rotation=o_rot,
                                        camera_rotation=cam_rot)

    @staticmethod
    def get_avatar_position(region: RegionBounds, offset: float = 0.0) -> np.array:
        """
        :param region: The scene region bounds.
        :param offset: Restrict the agent from offset to the edge of the region.

        :return: The position of the avatar for the next image as a numpy array.
        """
        if offset > 0.0:
            assert region.x_max - region.x_min > 2 * offset, "region x too small"
            x_min = region.x_min + offset
            x_max = region.x_max - offset
            assert region.z_max - region.z_min > 2 * offset, "region z too small"
            z_min = region.z_min + offset
            z_max = region.z_max - offset
            # assert region.y_max > 0.4 + offset, "region y too small"
            # y_max = region.y_max - offset
            y_max = region.y_max
            return np.array([RNG.uniform(x_min, x_max),
                             RNG.uniform(0.4, y_max),
                             RNG.uniform(z_min, z_max)])
        else:
            return np.array([RNG.uniform(region.x_min, region.x_max),
                             RNG.uniform(0.4, region.y_max),
                             RNG.uniform(region.z_min, region.z_max)])

    def get_object_position_commands(self, o_id: int, avatar_position: np.array, region: RegionBounds) -> List[dict]:
        """
        :param o_id: The object ID.
        :param avatar_position: The position of the avatar.
        :param region: The scene region bounds.

        :return: The position of the object for the next image as a numpy array.
        """

        # Get a random distance from the avatar.
        d = RNG.uniform(0.8, 3)
        # Get a random position for the object constrained to the environment bounds.
        o_p = ImageDataset.sample_spherical() * d
        # Clamp the y value to positive.
        o_p[1] = abs(o_p[1])
        o_p = avatar_position + o_p

        # Clamp the y value of the object.
        if o_p[1] > region.y_max:
            o_p[1] = region.y_max
        return [{"$type": "teleport_object",
                 "id": o_id,
                 "position": TDWUtils.array_to_vector3(o_p)}]

    def get_object_rotation_commands(self, o_id: int, o_name: str) -> List[dict]:
        """
        :param o_id: The object ID.
        :param o_name: The object name.

        :return: A list of commands to rotate the object.
        """

        # Add rotation commands. If we're clamping the rotation, rotate the object within +/- 30 degrees on each axis.
        if self.clamp_rotation:
            return [{"$type": "rotate_object_to",
                     "id": o_id,
                     "rotation": self.initial_rotations[o_name]},
                    {"$type": "rotate_object_by",
                     "id": o_id,
                     "angle": RNG.uniform(-30, 30),
                     "axis": "pitch"},
                    {"$type": "rotate_object_by",
                     "id": o_id,
                     "angle": RNG.uniform(-30, 30),
                     "axis": "yaw"},
                    {"$type": "rotate_object_by",
                     "id": o_id,
                     "angle": RNG.uniform(-30, 30),
                     "axis": "roll"}]
        # Set a totally random rotation.
        else:
            return [{"$type": "rotate_object_to",
                     "id": o_id,
                     "rotation": {"x": RNG.uniform(-360, 360),
                                  "y": RNG.uniform(-360, 360),
                                  "z": RNG.uniform(-360, 360),
                                  "w": RNG.uniform(-360, 360)}}]

    def get_camera_rotation_commands(self, o_id: int) -> List[dict]:
        """
        :param o_id: The object ID.

        :return: A list of commands to rotate the camera.
        """

        return [{"$type": "look_at",
                 "object_id": o_id,
                 "use_centroid": True},
                {"$type": "rotate_sensor_container_by",
                 "angle": RNG.uniform(-15, 15),
                 "axis": "pitch"},
                {"$type": "rotate_sensor_container_by",
                 "angle": RNG.uniform(-15, 15),
                 "axis": "yaw"}]

    @staticmethod
    def sample_spherical(npoints=1, ndim=3) -> np.array:
        vec = RNG.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        return np.array([vec[0][0], vec[1][0], vec[2][0]])

    def zip_images(self) -> None:
        """
        Zip up the images.
        """

        # Use a random token to avoid overwriting zip files.
        token = token_urlsafe(4)
        zip_path = self.output_directory.parent.joinpath(f"images_{token}.zip")

        # Source: https://thispointer.com/python-how-to-create-a-zip-archive-from-multiple-files-or-directory/
        with ZipFile(str(zip_path.resolve()), 'w') as zip_obj:
            # Iterate over all the files in directory
            for folderName, subfolders, filenames in os.walk(str(self.output_directory.resolve())):
                for filename in filenames:
                    if filename == '.DS_Store':
                        continue
                    # create complete filepath of file in directory
                    file_path = os.path.join(folderName, filename)
                    # Add file to zip
                    zip_obj.write(file_path, os.path.basename(file_path))
        # Remove the original images.
        dir_util.remove_tree(str(self.output_directory.resolve()))
