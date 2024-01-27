from typing import Dict


class ImagePosition:
    """
    The positions and rotations of the avatar and object for an image.

    Positions are stored as (x, y, z) dictionaries, for example: `{"x": 0, "y": 0, "z": 0}`.
    Rotations are stored as (x, y, z, w) dictionaries, for example: `{"x": 0, "y": 0, "z": 0, "w": 1}`.
    """

    def __init__(self, 
                 avatar_position: Dict[str, float],
                 camera_rotation: Dict[str, float],
                 object_position: Dict[str, float],
                 object_rotation: Dict[str, float],
                 object_center_position: Dict[str, float],
                 ):
        """
        :param avatar_position: The position of the avatar.
        :param camera_rotation: The rotation of the camera.
        :param object_position: The position of the object.
        :param object_rotation: The rotation of the object.
        :param object_center_position: The position of the center of the object.
        """
        self.avatar_position: Dict[str, float] = avatar_position
        self.camera_rotation: Dict[str, float] = camera_rotation
        self.object_position: Dict[str, float] = object_position
        self.object_rotation: Dict[str, float] = object_rotation
        self.object_center_position: Dict[str, float] = object_center_position
