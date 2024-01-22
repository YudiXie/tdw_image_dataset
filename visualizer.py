
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt


def load_image(image_filepath):
    """Load an image from disk and return a PIL.Image object.
    from https://github.com/brain-score/model-tools/blob/75365b54670d3f6f63dcdf88395c0a07d6b286fc/model_tools/activations/pytorch.py#L118
    """
    with Image.open(image_filepath) as pil_image:
        if 'L' not in pil_image.mode.upper() and 'A' not in pil_image.mode.upper() \
                and 'P' not in pil_image.mode.upper():  # not binary and not alpha and not palletized
            # work around to https://github.com/python-pillow/Pillow/issues/1144,
            # see https://stackoverflow.com/a/30376272/2225200
            return pil_image.copy()
        else:  # make sure potential binary images are in RGB
            rgb_image = Image.new("RGB", pil_image.size)
            rgb_image.paste(pil_image)
            return rgb_image


def show_image_and_meta(dset_path, index_row):
    image_idx, scene_n, wnid, model_n = index_row['image_index'], index_row['scene'], index_row['wnid'], index_row['model']
    headers = dset_path.joinpath('img_meta_headers.txt').read_text(encoding="utf-8").split("\n")
    img_path = dset_path.joinpath('images', scene_n, wnid, model_n, f"img_{image_idx:010d}.jpg")
    img_meta_path = dset_path.joinpath('images', scene_n, wnid, model_n, f"img_{image_idx:010d}_info.csv")

    img_meta = pd.read_csv(img_meta_path, names=headers).iloc[0]

    img = load_image(img_path)
    width, height = img.size

    print(f"image_idx: {image_idx}")
    print(f"scene: {scene_n}, wind: {wnid}")
    print(f"model: {model_n}")
    print(f'skybox: {img_meta["skybox_name"]}')

    print(f'screen_pos_x: {img_meta["ty"]:.2f}, screen_pos_y: {img_meta["tz"]:.2f}')
    print(f'screen_pos_x_frac: {(img_meta["ty"] / width + 0.5):.2f}, screen_pos_y_frac: {(img_meta["tz"] / height + 0.5):.2f}')
    print(f'screen_distance: {img_meta["neg_x"]:.2f}')
    print(f'relative Eular angles: {img_meta["euler_1"]:.2f}, {img_meta["euler_2"]:.2f}, {img_meta["euler_3"]:.2f}')

    print(f'object positions: x: {img_meta["object_pos_x"]:.2f}, y: {img_meta["object_pos_y"]:.2f}, z: {img_meta["object_pos_z"]:.2f}')
    print(f'object rotations: x: {img_meta["object_rot_x"]:.2f}, y: {img_meta["object_rot_y"]:.2f}, z: {img_meta["object_rot_z"]:.2f}, w: {img_meta["object_rot_w"]:.2f}')
    print(f'avator positions: x: {img_meta["avatar_pos_x"]:.2f}, y: {img_meta["avatar_pos_y"]:.2f}, z: {img_meta["avatar_pos_z"]:.2f}')
    print(f'camera rotations: x: {img_meta["camera_rot_x"]:.2f}, y: {img_meta["camera_rot_y"]:.2f}, z: {img_meta["camera_rot_z"]:.2f}, w: {img_meta["camera_rot_w"]:.2f}')

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.scatter(width // 2 + img_meta['ty'],
               height // 2 - img_meta['tz'],
               s=50, c='r', marker='x', label='positions')
