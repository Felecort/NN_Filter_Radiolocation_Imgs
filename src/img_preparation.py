from PIL import ImageOps, Image
import numpy as np
from os import listdir


def check_existing_datasets(dataset_name, datasets_path) -> bool:
    """This function check does the dataset exist.
    If True new dataset won't be created"""
    
    datasets = listdir(datasets_path)
    if dataset_name in datasets:
        return False
    return True


def add_noise(img, win_size, scale=0.2707)-> np.array:
    """Add the noise on image using
    (img + img * noise) formula"""
    
    # Create noise
    noise = np.random.rayleigh(scale=scale, size=(win_size, win_size))
    # Add noise
    noised_img = img + img * noise
    return noised_img


def add_borders(img, win_size, x, y) -> Image:
    """Create images with mirrowed and
    flipped borders with win_size // 2 width"""
    
    # Sides
    left_side = img.crop((0, 0, win_size, y))
    right_side = img.crop((x - win_size, 0, x, y))
    top_side = img.crop((0, 0, x, win_size))
    bottom_side = img.crop((0, y - win_size, x, y))

    # Flip or mirrowed sides
    rot_left_side = ImageOps.mirror(left_side)
    rot_right_side = ImageOps.mirror(right_side)
    rot_top_side = ImageOps.flip(top_side)
    rot_bottom_side = ImageOps.flip(bottom_side)

    # Corners
    top_left = left_side.crop((0, 0, win_size, win_size))
    top_right = right_side.crop((0, 0, win_size, win_size))
    bottom_left = left_side.crop((0, y - win_size, win_size, y))
    bottom_right = right_side.crop((0, y - win_size, win_size, y))

    # flipped and mirrowed corners
    rot_top_left = ImageOps.flip(ImageOps.mirror(top_left))
    rot_top_right = ImageOps.flip(ImageOps.mirror(top_right))
    rot_bottom_left = ImageOps.flip(ImageOps.mirror(bottom_left))
    rot_bottom_right = ImageOps.flip(ImageOps.mirror(bottom_right))

    # Create new image
    size = (x + 2 * win_size, y + 2 * win_size)
    new_image = Image.new("L", size=size)

    # Add corners
    new_image.paste(rot_top_left, (0, 0))
    new_image.paste(rot_top_right, (win_size + x, 0))
    new_image.paste(rot_bottom_left, (0, win_size + y))
    new_image.paste(rot_bottom_right, (x + win_size, y + win_size))

    # Add sides
    new_image.paste(rot_top_side, (win_size, 0))
    new_image.paste(rot_bottom_side, (win_size, win_size + y))
    new_image.paste(rot_left_side, (0, win_size))
    new_image.paste(rot_right_side, (x + win_size, win_size))

    # Add main path
    new_image.paste(img, (win_size, win_size))
    return new_image
