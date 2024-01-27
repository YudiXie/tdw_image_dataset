import os
import shutil
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
from tdw.output_data import OutputData, Occlusion, Images, ImageSensors, Transforms, Version
from tdw.librarian import ModelLibrarian, MaterialLibrarian, HDRISkyboxLibrarian, ModelRecord, HDRISkyboxRecord
from tdw.scene_data.scene_bounds import SceneBounds
from tdw.scene_data.region_bounds import RegionBounds
from tdw_image_dataset.image_position import ImagePosition
from tdw.quaternion_utils import QuaternionUtils

# The required version of TDW.
REQUIRED_TDW_VERSION: str = "1.9.0"
RNG: np.random.RandomState = np.random.RandomState(0)


def sample_spherical(npoints=1, ndim=3) -> np.array:
    """
    Generate a random point on the surface of a unit sphere.
    return the (x, y, z) coordinates of the point
    """
    vec = RNG.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return np.array([vec[0][0], vec[1][0], vec[2][0]])


def sample_spherical_cap(y_min=-0.2):
    while True:
        vec = sample_spherical()
        if vec[1] > y_min:
            break
    return vec


def sample_avatar_object_position(scene_bounds: SceneBounds, offset: float = 0.0, scene_name: str = '') -> np.array:
    """
    :param scene_bounds: The scene bounds.
    :param offset: Restrict the agent from offset to the edge of the region.

    :return: The position of the avatar and object for the next image as a numpy array.
    """
    # Get a random region within the scene.
    region: RegionBounds = scene_bounds.regions[RNG.randint(0, len(scene_bounds.regions))]
    
    y_min = 0.4

    if offset > 0.0:
        assert region.x_max - region.x_min > 2 * offset, "region x too small"
        x_min = region.x_min + offset
        x_max = region.x_max - offset
        assert region.z_max - region.z_min > 2 * offset, "region z too small"
        z_min = region.z_min + offset
        z_max = region.z_max - offset
        assert region.y_max > 0.4 + offset, "region y too small"
        y_max = region.y_max - offset        
    else:
        x_min, x_max = region.x_min, region.x_max
        z_min, z_max = region.z_min, region.z_max
        y_max = region.y_max

    obj_ymax = region.y_max
    
    if scene_name == 'savanna_flat_6km':
        # this scene is too big, need to clamp y range
        y_min, y_max = 5.0, 10.0
        obj_ymax = 15.0
    elif scene_name == 'suburb_scene_2023':
        x_min, x_max = -68.0, 68.0
        z_min, z_max = -11.0, 11.0
    elif scene_name == 'downtown_alleys':
        x_min, x_max = 29.0, 33.0
        z_min, z_max = -15.0, 8.0
        y_min, y_max = 2.0, 5.0
        obj_ymax = 7.0
    
    avatar_p = np.array([RNG.uniform(x_min, x_max),
                         RNG.uniform(y_min, y_max),
                         RNG.uniform(z_min, z_max)])
    
    resample_ct = 0
    resample_num = 15
    while resample_ct < resample_num:
        # Get a random distance from the avatar.
        distance = RNG.uniform(0.9, 4.2)
        # Get a random position for the object, not too low, and constrained to the environment bounds.
        object_offset = sample_spherical_cap() * distance
        object_p = avatar_p + object_offset
        if object_p[1] < obj_ymax:
            break
        resample_ct += 1

    return avatar_p, object_p


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
    EXCLUDE_MODELS = [
        'b02_bag', 'lantern_2010', 'b04_bottle_max', # models that have multiple objects are removed
        'heart', 'shark', # models that have wrong pivot points are removed
        ]
    # The headers of the metadata file.
    IMG_META_HEADERS = (
        'scene_name',
        'wnid',
        'record_wcategory',
        'record_name',
        'image_file_name',
        'skybox_name',
        'rel_pos_x', # left-right position of object center in camera reference frame, center of image is 0, + is going right, unit in 3D space in TDW
        'rel_pos_y', # up-down position of object center in camera reference frame, center of image is 0, + is going up, unit in 3D space in TDW
        'rel_pos_z', # distance of object center, camera is 0, + is going into the image, unit in 3D space in TDW
        'rel_rot_x', # rotation quaternion of object in camera reference frame
        'rel_rot_y',
        'rel_rot_z',
        'rel_rot_w',
        'rel_rot_euler_0', # rotation euler angles of object in camera reference frame
        'rel_rot_euler_1',
        'rel_rot_euler_2',
        'avatar_pos_x',
        'avatar_pos_y',
        'avatar_pos_z',
        'camera_rot_x',
        'camera_rot_y',
        'camera_rot_z',
        'camera_rot_w',
        'object_pos_x', # object center position in world space
        'object_pos_y',
        'object_pos_z',
        'object_rot_x',
        'object_rot_y',
        'object_rot_z',
        'object_rot_w',
    )

    def __init__(self,
                 output_directory: Union[str, Path],
                 port: int = 1071,
                 launch_build: bool = True,
                 materials: bool = False,
                 screen_width: int = 256,
                 screen_height: int = 256,
                 output_scale: float = 1,
                 hdri: bool = True,
                 show_objects: bool = True,
                 clamp_rotation: bool = True,
                 max_height: float = 0.5,
                 occl_filter_th: float = 0.3,
                 less_dark: bool = True,
                 exterior_only: bool = True,
                 id_pass: bool = False,
                 num_img_total: int = 5000,
                 library: str = "models_core.json",
                 random_seed: int = 0,
                 subset_wnids: Optional[List[str]] = None,
                 offset: float = 0.3,
                 scene_list = ['building_site', ],
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
        :param occl_filter_th: The occlusion filtering threshold. Lower value = slower FPS, better composition. Must be between 0 and 1.
        :param less_dark: If True, there will be more daylight exterior skyboxes (requires hdri == True)
        :param exterior_only: If True, only use exterior skyboxes (requires hdri == True)
        :param id_pass: If True, send and save the _id pass.
        :param num_img_total: The number of generated images for all scenes,
            the generated image number will be close to this number but not exactly the same
            to ensure that each model in the wnid category has the same number of images
        :param library: The path to the library records file.
        :param random_seed: The random seed.
        :param subset_wnids: create a subset only use these wnid categories.
        :param offset: Restrict the agent from offset to the edge of the region.
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
        self.occl_filter_th: float = occl_filter_th
        """:field
        If True, send and save the _id pass.
        """
        self.id_pass: bool = id_pass
        """:field
        Restrict the agent from offset to the edge of the region.
        """
        self.offset = offset

        self.subset_wnids = subset_wnids
        self.current_scene = ''
        self.scene_list = scene_list

        assert 0 < max_height <= 1.0, f"Invalid max height: {max_height}"
        assert 0 < occl_filter_th <= 1.0, f"Invalid occlusion threshold: {occl_filter_th}"

        """:field
        If True, there will be more daylight exterior skyboxes (requires hdri == True)
        """
        self.less_dark: bool = less_dark
        """:field
        If True, only use exterior skyboxes (requires hdri == True)
        """
        self.exterior_only: bool = exterior_only
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
        super().__init__(port=port, launch_build=launch_build, check_version=True)
        resp = self.communicate({"$type": "send_version"})
        for i in range(len(resp) - 1):
            if OutputData.get_data_type_id(resp[i]) == "vers":
                build_version = Version(resp[i]).get_tdw_version()
                print(f"Build version: {build_version}, required version: >={REQUIRED_TDW_VERSION}")

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
            if self.exterior_only:
                self.skyboxes = [s for s in self.skyboxes if s.location == "exterior"]
            # Prefer exterior daytime skyboxes by adding them multiple times to the list.
            if self.less_dark:
                temp = self.skyboxes[:]
                for skybox in temp:
                    if skybox.location == "exterior" and skybox.sun_elevation >= 145:
                        self.skyboxes.append(skybox)
        
        self.wnid2models = {}
        if self.subset_wnids:
            # Fetch the WordNet IDs from the given subset
            wnids_list = self.subset_wnids
            for w in wnids_list:
                wnid_models_raw = Controller.MODEL_LIBRARIANS[self.model_library_file].get_all_models_in_wnid(w)
                wnid_models = [r for r in wnid_models_raw if ((not r.do_not_use) and (r.name not in self.EXCLUDE_MODELS) and (not r.composite_object))]
                assert len(wnid_models) > 0, f"ID: {w} do not have usable models"
                self.wnid2models[w] = wnid_models
        else:
            # Fetch the WordNet IDs.
            wnids_list = Controller.MODEL_LIBRARIANS[self.model_library_file].get_model_wnids()
            # Remove any wnids in wnids_list that don't have valid models.
            for w in wnids_list:
                wnid_models_raw = Controller.MODEL_LIBRARIANS[self.model_library_file].get_all_models_in_wnid(w)
                wnid_models = [r for r in wnid_models_raw if ((not r.do_not_use) and (r.name not in self.EXCLUDE_MODELS) and (not r.composite_object))]
                if len(wnid_models) > 0:
                    self.wnid2models[w] = wnid_models
        
        # list of all usable object catoegories wnids
        self.wnids = list(self.wnid2models.keys())

        # equal number of objects per scene, per category (wnid), but each wind has different number of models
        num_img_per_scene = num_img_total / len(self.scene_list) # ~1.1M, if total num_img_total = 10M, 9 scenes
        num_img_per_wnid = num_img_per_scene / len(self.wnids) # ~8680, if there are 128 wnids

        # round the numbers so that all models in each wnid has the same number of images
        self.wnid2num_img_per_model = {}
        img_per_scene_round = 0
        for w in self.wnids:
            image_num = round(num_img_per_wnid / len(self.wnid2models[w])) # ~8680 if 1 model, ~4340 if 2 models
            self.wnid2num_img_per_model[w] = image_num
            img_per_scene_round += image_num * len(self.wnid2models[w])

        self.num_img_per_scene = img_per_scene_round
        self.num_img_total = self.num_img_per_scene * len(self.scene_list)

        self.generate_index()

        # log dataset meta data
        self.generate_metadata()

    def generate_index(self) -> None:
        """
        Generate a index file for this dataset.
        The generated index has self.num_img_total rows, each row is an image
        each scence has equal number of images, self.num_img_per_scene
        each wnid has roughly equal number of images, self.wnid2num_img_per_model[w] * len(self.wnid2models[w])
        each model under the same wnid has the same number of images, self.wnid2num_img_per_model[w]
        models under different wnids have different number of images
        """
        self.index_file_n = str(self.output_directory.joinpath(f'index_img_{self.num_img_total}.csv').resolve())
        if os.path.exists(self.index_file_n):
            print(f"index file {self.index_file_n} already exists, skip generating index file")
        else:
            scene_col = []
            for scene in self.scene_list:
                scene_col.extend([scene, ] * self.num_img_per_scene)
            
            wnid_col_per_scene = []
            model_col_per_scene = []
            for w in self.wnids:
                wnid_col_per_scene.extend([w, ] * (self.wnid2num_img_per_model[w] * len(self.wnid2models[w])))
                for r in self.wnid2models[w]:
                    model_col_per_scene.extend([r.name, ] * self.wnid2num_img_per_model[w])
            wnid_col = wnid_col_per_scene * len(self.scene_list)
            model_col = model_col_per_scene * len(self.scene_list)

            assert len(scene_col) == len(wnid_col) == len(model_col) == self.num_img_total

            index_df = pd.DataFrame({
                'scene': scene_col,
                'wnid': wnid_col,
                'model': model_col,
            })
            index_df.to_csv(self.index_file_n)

    def generate_metadata(self) -> None:
        """
        Generate a metadata file for this dataset.
        """
        data = {"dataset": str(self.output_directory.resolve()),
                "scene_list": self.scene_list,
                "num_img_total": self.num_img_total,
                "materials": self.materials,
                "hdri": self.skyboxes is not None,
                "screen_width": self.screen_width,
                "screen_height": self.screen_height,
                "output_scale": self.scale,
                "clamp_rotation": self.clamp_rotation,
                "show_objects": self.show_objects,
                "max_height": self.max_height,
                "occl_filter_th": self.occl_filter_th,
                "less_dark": self.less_dark,
                "exterior_only": self.exterior_only,
                "start": datetime.now().strftime("%H:%M %d.%m.%y")}
        self.metadata_path.write_text(json.dumps(data, sort_keys=True, indent=4), encoding="utf-8")

        # save the meta headers into a file
        img_meta_headers_path = self.output_directory.joinpath(f"img_meta_headers.txt")
        img_meta_headers_path.write_text("\n".join(self.IMG_META_HEADERS), encoding="utf-8")
    
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

    def generate_multi_scene(self, do_zip=True) -> None:

        done_scenes_path: Path = self.output_directory.joinpath(f"processed_scenes.txt")
        processed_scenes_names: List[str] = []
        if done_scenes_path.exists():
            processed_scenes_names = done_scenes_path.read_text(encoding="utf-8").split("\n")
        
        num_scene = len(self.scene_list)
        for i, scene_n in enumerate(self.scene_list):
            if scene_n in processed_scenes_names:
                print(f"Scene: {scene_n} already processed, skip")
                continue

            print(f"Generating: {scene_n}\t{i + 1}/{num_scene}")
            self.generate_single_scene(scene_name=scene_n)

            if do_zip:
                scene_path = self.images_directory.joinpath(scene_n)
                shutil.make_archive(scene_path, 'zip', scene_path) # below python 3.10.6, maynot be thread-safe
                shutil.rmtree(scene_path.resolve())
            
            # Mark this scene as processed.
            with done_scenes_path.open("at", encoding="utf-8") as f:
                f.write(f"\n{scene_n}")
        
        self.communicate({"$type": "terminate"})


    def generate_single_scene(self, scene_name: str) -> None:
        """
        Generate the dataset for a single scene
        :param scene_name: The scene name.
        """

        scene_num = self.scene_list.index(scene_name)
        # read index for this scene, skip the first row which contains the column names
        self.scene_index = pd.read_csv(self.index_file_n,
                                       names=['scene', 'wnid', 'model'],
                                       index_col=0,
                                       skiprows=1 + scene_num * self.num_img_per_scene, 
                                       nrows=self.num_img_per_scene)
        # check if the index is correct
        unique_scenes = self.scene_index['scene'].unique()
        assert len(unique_scenes) == 1 and unique_scenes[0] == scene_name, "scene name mismatch"

        # Initialize the scene.
        scene_bounds: SceneBounds = self.initialize_scene(scene_name)

        # Create the progress bar.
        pbar = tqdm(total=len(self.wnids))

        done_models_path: Path = self.output_directory.joinpath(f"{scene_name}_processed_records.txt")
        # Get a list of models that have already been processed.
        processed_model_names: List[str] = []
        if done_models_path.exists():
            processed_model_names = done_models_path.read_text(encoding="utf-8").split("\n")

        # Iterate through each wnid.
        for w in self.wnids:
            # Update the progress bar.
            pbar.set_description(w)
            self.wnid_index = self.scene_index[self.scene_index['wnid'] == w]

            # Get all valid models in the wnid.
            records = self.wnid2models[w]

            # Process each record.
            fps = "nan"
            for record, i in zip(records, range(len(records))):
                # Set the progress bar description to the wnid and FPS.
                pbar.set_description(f"record {i + 1}/{len(records)}, FPS {fps}")

                # Skip models that have already been processed.
                if record.name in processed_model_names:
                    continue

                # Create all of the images for this model.
                fps = self.process_model(record, scene_bounds, w)

                # Mark this record as processed.
                with done_models_path.open("at", encoding="utf-8") as f:
                    f.write(f"\n{record.name}")
            pbar.update(1)
        pbar.close()

        # Add the end time to the metadata file.
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        end_time = datetime.now().strftime("%H:%M %d.%m.%y")
        metadata.update({"end": end_time})
        metadata['scene_list'].append(self.current_scene)
        self.metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=4), encoding="utf-8")

        # Don't need to unload the scene here since loading a new scene 
        # will automatically unload the old one, should doulbe check this

    def process_model(self, record: ModelRecord, scene_bounds: SceneBounds, wnid: str) -> float:
        """
        Capture images of a model.

        :param record: The model record.
        :param scene_bounds: The bounds of the scene.
        :param wnid: The wnid of the record.
        :return The rendering fps for the current model.
        """
        # create folder for this model
        out_path = self.images_directory.joinpath(self.current_scene).joinpath(wnid).joinpath(record.name)
        out_path.mkdir(parents=True, exist_ok=True)

        self.model_index = self.wnid_index[self.wnid_index['model'] == record.name]
        img_count_per_model = len(self.model_index)
        assert img_count_per_model == self.wnid2num_img_per_model[wnid], "image count mismatch"

        # the first index of images generated for this model in this scene
        assert self.model_index.index[-1] - self.model_index.index[0] + 1 == img_count_per_model, 'indexes should be squential'
        image_index = self.model_index.index[0]

        image_positions: List[ImagePosition] = []
        o_id = self.get_unique_id()

        # Add the object.
        resp = self.communicate(self.get_object_initialization_commands(record=record, o_id=o_id))
        
        o_init_transforms = None
        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "tran":
                o_init_transforms = Transforms(resp[i])
                assert o_init_transforms.get_num() == 1, "object transform should only have one object"
        
        o_init_rot = o_init_transforms.get_rotation(0)
        # Cache the initial rotation of the object.
        if record.name not in self.initial_rotations:
            self.initial_rotations[record.name] = TDWUtils.array_to_vector4(o_init_rot)
        
        # The index in the HDRI records array.
        self.skybox_idx = 0
        # The count of images on this skybox so far.
        self.skybox_img_idx = 0
        skybox_name = 'initial'
        if self.skyboxes:
            # The number of images per skybox for this model in this scene.
            self.imgs_per_skybox = int(img_count_per_model / len(self.skyboxes))
            if self.imgs_per_skybox == 0:
                self.imgs_per_skybox = 1

        while len(image_positions) < img_count_per_model:
            # sample configureation and get the occlusion values
            v_occl, v_unoccl, image_position = self.sample_configuration(o_id, scene_bounds)
            occl_frac = 1 - v_occl / (v_unoccl + 0.001)  # fraction of the object occluded by scenes
            if (occl_frac < self.occl_filter_th) and (v_occl > 2):
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

        for p in image_positions:
            # Teleport the avatar.
            # Rotate the avatar's camera.
            # Teleport the object.
            # Rotate the object.
            # Get the response.
            commands = [{"$type": "teleport_avatar_to",
                         "position": p.avatar_position,
                         "avatar_id": ImageDataset.AVATAR_ID,
                         },
                        {"$type": "rotate_sensor_container_to",  # will not change avator position
                         "rotation": p.camera_rotation,
                         "avatar_id": ImageDataset.AVATAR_ID,
                         },
                        {"$type": "teleport_object",
                         "id": o_id,
                         "position": p.object_position,
                         },
                        {"$type": "rotate_object_to",  # will not change object pivot position
                         "id": o_id,
                         "rotation": p.object_rotation,
                         },]

            # Set the visual materials.
            if self.materials:
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
            
            # Maybe set a new skybox. Rotate the skybox.
            if self.skyboxes:
                # the name of the skybox the following command set to
                skybox_name = self.skyboxes[self.skybox_idx].name
                # Set a new skybox.
                if self.skybox_img_idx == 0:
                    command = self.get_add_hdri_skybox(self.skyboxes[self.skybox_idx].name)
                # It's not time yet to set a new skybox. Don't send a command.
                else:
                    command = None
                                
                if command:
                    commands.append(command)
                # commands.append({"$type": "rotate_hdri_skybox_by",
                #                  "angle": RNG.uniform(0, 360)})
                
                self.skybox_img_idx += 1
                if self.skybox_img_idx >= self.imgs_per_skybox:
                    self.skybox_img_idx = 0
                    # move to the next skybox in the next call
                    self.skybox_idx += 1
                    if self.skybox_idx >= len(self.skyboxes):
                        self.skybox_idx = 0

            img_resp = self.communicate(commands)

            # get the relative position and rotation of the object center in camera reference frame
            rel_pos = QuaternionUtils.world_to_local_vector(TDWUtils.vector3_to_array(p.object_center_position),
                                                            TDWUtils.vector3_to_array(p.avatar_position),
                                                            TDWUtils.vector4_to_array(p.camera_rotation))
            rel_rot = QuaternionUtils.multiply(QuaternionUtils.get_inverse(TDWUtils.vector4_to_array(p.camera_rotation)),
                                               TDWUtils.vector4_to_array(p.object_rotation))
            rel_rot_euler = QuaternionUtils.quaternion_to_euler_angles(rel_rot)

            save_tuple = (
                self.current_scene,
                wnid,
                record.wcategory,
                record.name,
                f"img_{image_index:010d}",
                skybox_name,
                rel_pos[0],
                rel_pos[1],
                rel_pos[2],
                rel_rot[0],
                rel_rot[1],
                rel_rot[2],
                rel_rot[3],
                rel_rot_euler[0],
                rel_rot_euler[1],
                rel_rot_euler[2],
                p.avatar_position['x'],
                p.avatar_position['y'],
                p.avatar_position['z'],
                p.camera_rotation['x'],
                p.camera_rotation['y'],
                p.camera_rotation['z'],
                p.camera_rotation['w'],
                p.object_center_position['x'],
                p.object_center_position['y'],
                p.object_center_position['z'],
                p.object_rotation['x'],
                p.object_rotation['y'],
                p.object_rotation['z'],
                p.object_rotation['w'],
            )

            # Create a thread to save the image.
            t = Thread(target=self.save_image, args=(img_resp, save_tuple, out_path, image_index))
            t.daemon = True
            t.start()

            image_index += 1
            
        t1 = time()

        # Stop sending images.
        # Destroy the object.
        # Unload asset bundles.
        self.communicate([{"$type": "send_images",
                           "frequency": "never"},
                          {"$type": "destroy_object",
                           "id": o_id},
                          {"$type": "unload_asset_bundles"}])
        return round(img_count_per_model / (t1 - t0))

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

    def save_image(self, resp, save_tuple: tuple, output_directory: Path, image_index: int) -> None:
        """
        Save an image.

        :param resp: The raw response data.
        :param save_tuple: The metadata to save.
        :param record: The model record.
        :param image_index: The image index.
        :param wnid: The wnid.
        """
        # Save the image.
        filename = f"{image_index:010d}"
        
        assert len(save_tuple) == len(self.IMG_META_HEADERS), "save tuple length mismatch"
        save_dict = {k: [v, ] for k, v in zip(self.IMG_META_HEADERS, save_tuple)}
        csv_path = output_directory.joinpath("img_" + filename + "_info.csv")
        pd.DataFrame.from_dict(save_dict).to_csv(csv_path, header=False, index=False)

        # Save the image without resizing.
        if not self.scale:
            TDWUtils.save_images(Images(resp[0]), filename,
                                 output_directory=output_directory)
        # Resize the image and save it.
        else:
            TDWUtils.save_images(Images(resp[0]), filename,
                                 output_directory=output_directory,
                                 resize_to=self.output_size)

    def sample_configuration(self, o_id: int, scene_bounds: SceneBounds) -> Tuple[int, int, ImagePosition]:
        """
        Sample a configuration for object positioin, rotation, camera position and rotation

        :param o_id: The ID of the object.
        :param scene_bounds: The scene bounds to sample from.
        """
        # Get a random position for the avatar.
        a_p, o_center_p = sample_avatar_object_position(scene_bounds, self.offset, self.current_scene)
        a_p = TDWUtils.array_to_vector3(a_p)
        o_center_p = TDWUtils.array_to_vector3(o_center_p)

        commands = [{"$type": "teleport_object",
                     "id": o_id,
                     "position": o_center_p,
                     },
                    {"$type": "teleport_avatar_to",
                     "position": a_p,
                     "avatar_id": ImageDataset.AVATAR_ID,
                     },
                     ]

        # Rotate the object. 
        # If we're clamping the rotation, rotate the object within +/- rot_range degrees on each axis.
        # all of these will not change object poviot (bottom center) position, but will change object centorid position
        if self.clamp_rotation:
            rot_range = 45
            commands.extend([
                # an alternative to look at the camera is to initlize object by its initial rotation
                # {
                # "$type": "rotate_object_to",
                # "id": o_id,
                # "rotation": self.initial_rotations[o_name],
                # },
                {
                "$type": "object_look_at_position",
                "position": a_p,
                "id": o_id,
                },
                {
                "$type": "rotate_object_by",
                "id": o_id,
                "angle": RNG.uniform(-rot_range, rot_range),
                "axis": "pitch",
                "is_world": False,
                },
                {
                "$type": "rotate_object_by",
                "id": o_id,
                "angle": RNG.uniform(-rot_range, rot_range),
                "axis": "yaw",
                "is_world": False,
                },
                {
                "$type": "rotate_object_by",
                "id": o_id,
                "angle": RNG.uniform(-rot_range, rot_range),
                "axis": "roll",
                "is_world": False,
                },
            ])
        else:
        # Set a totally random rotation.
            commands.extend([
                {
                "$type": "rotate_object_to",
                "id": o_id,
                "rotation": {"x": RNG.uniform(-360, 360),
                             "y": RNG.uniform(-360, 360),
                             "z": RNG.uniform(-360, 360),
                             "w": RNG.uniform(-360, 360)},
                },
            ])
        
        # after rotating, set the object's centroid to the sampled position
        commands.extend([
            {"$type": "teleport_object",
            "id": o_id,
            "position": o_center_p,
            "use_centroid": True,
            },
        ])

        # Rotate the camera, all of these will not change avator position
        cam_rot_range = 20
        commands.extend([
            {"$type": "look_at",
             "object_id": o_id,
             "use_centroid": True,
             "avatar_id": ImageDataset.AVATAR_ID,
             },
            {"$type": "rotate_sensor_container_by",
             "angle": RNG.uniform(-cam_rot_range, cam_rot_range),
             "axis": "pitch",
             "avatar_id": ImageDataset.AVATAR_ID,
             },
            {"$type": "rotate_sensor_container_by",
             "angle": RNG.uniform(-cam_rot_range, cam_rot_range),
             "axis": "yaw",
             "avatar_id": ImageDataset.AVATAR_ID,
             },
            ])
        
        # Request output data.
        commands.extend([{"$type": "send_occlusion",
                          "ids": [ImageDataset.AVATAR_ID, ],
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
        occl = None
        # a_p is not changed so we don't need to update it
        cam_rot = None
        o_rot = None
        o_p = None
        has_occl, has_imse, has_tran = False, False, False
        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "occl":
                occl = Occlusion(resp[i])
                v_occluded = occl.get_occluded()
                v_unoccluded = occl.get_unoccluded()
                has_occl = True
            elif r_id == "imse":
                img_sen = ImageSensors(resp[i])
                assert img_sen.get_num_sensors() == 1, "only one sensor"
                cam_rot = img_sen.get_sensor_rotation(0)
                cam_rot = {"x": cam_rot[0], "y": cam_rot[1], "z": cam_rot[2], "w": cam_rot[3]}
                has_imse = True
            elif r_id == "tran":
                transforms = Transforms(resp[i])
                assert transforms.get_num() == 1, "only one object"
                o_rot = TDWUtils.array_to_vector4(transforms.get_rotation(0))
                o_p = TDWUtils.array_to_vector3(transforms.get_position(0))
                has_tran = True
        assert has_occl and has_imse and has_tran, "missing occlusion, image sensor or transform"
        return v_occluded, v_unoccluded, ImagePosition(avatar_position=a_p,
                                                       object_position=o_p,
                                                       object_rotation=o_rot,
                                                       camera_rotation=cam_rot,
                                                       object_center_position=o_center_p)

    @staticmethod
    def zip_images(output_directory: Path) -> None:
        """
        Zip up the images.
        """

        # Use a random token to avoid overwriting zip files.
        token = token_urlsafe(4)
        zip_path = output_directory.parent.joinpath(f"images_{token}.zip")

        # Source: https://thispointer.com/python-how-to-create-a-zip-archive-from-multiple-files-or-directory/
        with ZipFile(str(zip_path.resolve()), 'w') as zip_obj:
            # Iterate over all the files in directory
            for folderName, subfolders, filenames in os.walk(str(output_directory.resolve())):
                for filename in filenames:
                    if filename == '.DS_Store':
                        continue
                    # create complete filepath of file in directory
                    file_path = os.path.join(folderName, filename)
                    # Add file to zip
                    zip_obj.write(file_path, os.path.basename(file_path))
        # Remove the original images.
        dir_util.remove_tree(str(output_directory.resolve()))
