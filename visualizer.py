
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

def load_image_and_meta(dset_path, index_row):
    image_idx, scene_n, wnid, model_n = index_row['image_index'], index_row['scene'], index_row['wnid'], index_row['model']
    headers = dset_path.joinpath('img_meta_headers.txt').read_text(encoding="utf-8").split("\n")
    img_path = dset_path.joinpath('images', scene_n, wnid, model_n, f"img_{image_idx:010d}.jpg")
    img_meta_path = dset_path.joinpath('images', scene_n, wnid, model_n, f"img_{image_idx:010d}_info.csv")

    img_meta = pd.read_csv(img_meta_path, names=headers).iloc[0]
    img = load_image(img_path)
    return image_idx, img, img_meta


def show_image_and_meta(img, img_meta):
    width, height = img.size
    
    print(f"scene: {img_meta['scene_name']}, wind: {img_meta['wnid']}")
    print(f'category: {img_meta["record_wcategory"]}')
    print(f"model: {img_meta['record_name']}")
    print(f'skybox: {img_meta["skybox_name"]}')

    print(f'screen_pos_x: {img_meta["rel_pos_x"]:.2f}, screen_pos_y: {img_meta["rel_pos_y"]:.2f}')
    print(f'screen_distance: {img_meta["rel_pos_z"]:.2f}')
    print(f'relative Eular angles: {img_meta["rel_rot_euler_0"]:.2f}, {img_meta["rel_rot_euler_1"]:.2f}, {img_meta["rel_rot_euler_2"]:.2f}')

    print(f'object positions: x: {img_meta["object_pos_x"]:.2f}, y: {img_meta["object_pos_y"]:.2f}, z: {img_meta["object_pos_z"]:.2f}')
    print(f'object rotations: x: {img_meta["object_rot_x"]:.2f}, y: {img_meta["object_rot_y"]:.2f}, z: {img_meta["object_rot_z"]:.2f}, w: {img_meta["object_rot_w"]:.2f}')
    print(f'avator positions: x: {img_meta["avatar_pos_x"]:.2f}, y: {img_meta["avatar_pos_y"]:.2f}, z: {img_meta["avatar_pos_z"]:.2f}')
    print(f'camera rotations: x: {img_meta["camera_rot_x"]:.2f}, y: {img_meta["camera_rot_y"]:.2f}, z: {img_meta["camera_rot_z"]:.2f}, w: {img_meta["camera_rot_w"]:.2f}')

    fig, ax = plt.subplots()
    ax.imshow(img)
    scale_factor = 200 / img_meta["rel_pos_z"]  # 200 is just a guess now
    ax.scatter(width // 2 + img_meta['rel_pos_x'] * scale_factor,
               height // 2 - img_meta['rel_pos_y'] * scale_factor,
               s=50, c='r', marker='x', label='positions')
