from time import time
from os import listdir
from PIL import Image, ImageOps
import numpy as np
from torch import Tensor
import pathlib
from tqdm import tqdm
from image_processing import add_borders


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


def filtering_image(model, path_to_image, image_name, win_size, device):
    path = f"{path_to_image}\{image_name}"
    img = Image.open(path)
    shape = img.size
    out_arr = np.empty(shape)
    img = np.array(add_borders(img, win_size)) / 255
    for y in tqdm(range(shape[1])):
        for x in range(shape[0]):
            croped_img = img[y:win_size+y, x:x+win_size]
            data = Tensor([croped_img.flatten()]).float()
            data = data.to(device=device)
            res = model(data)
            out_arr[y, x] = res
    out_arr *= 255
    out = np.where(out_arr >= 255, 255, out_arr)
    out_img = Image.fromarray(out)
    out_path = fr"D:\Projects\PythonProjects\NIR\datasets\filtered_imgs\{image_name}"
    out_img.save(out_path)