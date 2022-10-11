from time import time
from os import listdir
from PIL import Image, ImageOps
import numpy as np
from torch import Tensor, no_grad
import pathlib
from tqdm import tqdm
from image_processing import add_borders
from skimage.metrics import structural_similarity as ssim

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
        img.save(f"{root_folder}\gray_images\{image_name}")


def get_batch(y, img, left, win_size, device):
    res = np.empty((304, 169))
    if left:
        for x in range(0, 304):
            res[x] = img[y:win_size+y, x:x+win_size].flatten()
    else:
        for i, x in enumerate(range(304, 608)):
            res[i] = img[y:win_size+y, x:x+win_size].flatten()
    return Tensor(res).float().to(device=device)


def filtering_image(model, path_to_image, image_name, win_size, device):
    out_path = fr"D:\Projects\PythonProjects\NIR\datasets\filtered_imgs\{image_name}"
    path = f"{path_to_image}\{image_name}"
    img = Image.open(path)
    shape = img.size
    out_arr = np.empty(shape)
    img = np.array(add_borders(img, win_size)) / 255
    model.eval()
    with no_grad():
        for y in tqdm(range(shape[1])):
            res = model(get_batch(y, img, 1, win_size, device)).cpu()
            out_arr[y, :304] = np.squeeze(np.array(res))
            res = model(get_batch(y, img, 0, win_size, device)).cpu()
            out_arr[y, 304:] = np.squeeze(np.array(res))
    out_arr *= 255
    out = np.where(out_arr >= 255, 255, out_arr)
    out = out.astype(np.uint8)
    out = Image.fromarray(out)
    out.save(out_path)
    
    
def check_ssim(filtered_images, genuine_images):
    filtered_imgs_list = listdir(filtered_images)
    for image_name in filtered_imgs_list:
        filtered_img = np.array(Image.open(f"{filtered_images}\{image_name}"))
        genuine_img = np.array(Image.open(f"{genuine_images}\{image_name}"))
        ssim_metric = ssim(filtered_img, genuine_img)
        print(f"{image_name}, SSIM = {ssim_metric}")
        