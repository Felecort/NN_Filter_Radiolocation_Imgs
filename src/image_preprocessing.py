from pprint import pprint
from traceback import print_tb
import numpy as np
from random import shuffle
from PIL import ImageOps, Image
from operator import mul
from functools import reduce

from pprint import pprint

##################################################
# Testing
def get_shuffled_idxs(img_path, img_list, step) -> list[list]:
    """
    Finds and returns the shuffled indexes of each image 
    """
    
    
    
    # path = f"{img_path}\{img}"
    # img_path = r"D:\Projects\PythonProjects\NIR\datasets\tmp"
    # img_list = listdir(img_path)
    # print(img_list)
    # x, y = Image.open(f"{img_path}\{img_list[0]}").size
    # print(x, y)
    # x_scaled, y_scaled = x // step, y // step

    # [[key1, [val1, val1]], [key2, [val1, val2]]]
    for img in img_list:
        img_size: tuple = Image.open(f"{img_path}\{img}").size
        x, y = [i // step for i in img_size]
        # print(x, y)
        keys = np.arange(y + 1) * 5
        vals = np.arange(x + 1) * 5
        # print(keys, vals)
        d = {key: vals.copy() for key in keys}
        d[35][3] = -1
        d[40][4]
        pprint(d)
        return 
        
        
        
    
    # pixels_in_imgs = [reduce(mul, Image.open(f"{img_path}\{img}").size) for img in img_list]
    # print(pixels_in_imgs)
    # idx_arrs = [list(range(0, idxs, step)) for idxs in pixels_in_imgs]
    
    # for arr in idx_arrs:
    #     shuffle(arr)
        
    # print(f"Indexes in first img {idx_arrs[0][:5]}")
    # print(f"Total indexes in ")
    return
##################################################

if __name__ == "__main__":
    from os import listdir
    img_path = r"D:\Projects\PythonProjects\NIR\datasets\images"
    imgs_list = listdir(img_path)
    step = 5
    
    get_shuffled_idxs(img_path, imgs_list, step)
    












def add_noise(img, win_size, scale=0.2707)-> np.array:
    """
    Adds the noise on image using
    (img + img * noise) formula
    """
    # Create noise
    noise = np.random.rayleigh(scale=scale, size=(win_size, win_size))
    # Add noise
    noised_img = img + img * noise
    return noised_img


def add_borders(img, win_size, x, y) -> Image:
    """
    Creates images with mirrowed and
    flipped borders with width = win_size // 2 
    """
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
