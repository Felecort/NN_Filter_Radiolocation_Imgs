from time import time
from os import listdir
from PIL import Image, ImageOps
import pathlib


def delta_time() -> 'function':
    start = time()

    def wrapper() -> float:
        return time() - start
    return wrapper


def convert_to_grayscale(path_to_images):
    p = pathlib.PureWindowsPath(path_to_images)
    images_list = listdir(p)
    root_folder = p.parent
    for image_name in images_list:
        path = f"{path_to_images}\{image_name}"
        img = ImageOps.grayscale(Image.open(path))
        img.save(f"{root_folder}\gray_images\gs_{image_name}")
