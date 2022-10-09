import numpy as np
from random import sample
from PIL import ImageOps, Image


def load_images(img_path, list_of_img_names) -> list:
    """
    Load all the images in determine folder and convert to grayscale
    """
    imgs_list = []
    for image_name in list_of_img_names:
        img = Image.open(f"{img_path}\{image_name}")
        img = ImageOps.grayscale(img)
        imgs_list.append(img)
    return imgs_list


def get_shuffled_idxs(*, imgs_list, step=1) -> list[dict]:
    """
    Finds and returns the shuffled pixel of image
    [{column1: [row1],
     column2: [row2],
     ...},
     ...]
    """
    shuffled_imgs_idxs = []
    keys_list = []
    for img in imgs_list:
        img_size = img.size  # tuple (x, y)
        x, y = (i // step for i in img_size)
        keys_y = list(range(0, (y + 1) * step, step))
        vals_x = list(range(0, (x + 1) * step, step))

        img_indxs = {key: sample(vals_x.copy(), x) for key in keys_y}
        keys_list.append(keys_y)
        shuffled_imgs_idxs.append(img_indxs)
    return shuffled_imgs_idxs, keys_list


def add_noise(img, scale=0.2707) -> np.array:
    """
    Adds the noise on image using
    (img + img * noise) formula
    """
    # Create noise
    size = img.shape
    noise = np.random.rayleigh(scale=scale, size=size)
    # Add noise
    noised_img = img + img * noise
    return noised_img


def add_borders(img, win_size) -> Image:
    """
    Creates images with mirrowed and
    flipped borders with width = win_size // 2 
    """
    x = img.size[0]
    y = img.size[1]

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
