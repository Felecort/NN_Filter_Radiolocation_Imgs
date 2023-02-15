from time import time
from os import listdir
from PIL import Image, ImageOps
import numpy as np
from torch import Tensor, no_grad
import pathlib
from tqdm import tqdm
from image_processing import add_borders
from skimage.metrics import structural_similarity as ssim
# from __future__ import division
# from scipy import ndimage
from scipy import signal

def delta_time() -> 'function':
    start = time()

    def wrapper() -> float:
        return time() - start
    return wrapper


def convert_to_grayscale(path_to_images) -> None:
    p = pathlib.PureWindowsPath(path_to_images)
    images_list = listdir(p)
    root_folder = p.parent
    for image_name in images_list:
        path = f"{path_to_images}\{image_name}"
        img = ImageOps.grayscale(Image.open(path))
        img.save(f"{root_folder}\gray_images\{image_name}")


def filtering_image(model, out_path, path_to_image, image_name, win_size, device, slices=0) -> None:

    out_path = out_path / image_name
    path = path_to_image / image_name

    img = Image.open(path)
    shape = img.size

    img = np.array(add_borders(img, win_size // 2)) / 255
    out_image = np.empty((shape[1], shape[0]))

    model.eval()
    with no_grad():
        if slices <= 1:
            for y in tqdm(range(shape[1])):
                raw_res = np.empty((shape[0], win_size ** 2))
                for x in range(shape[0]):
                    raw_res[x] = img[y:y+win_size, x:x+win_size].flatten()
                res = Tensor(raw_res).float().to(device=device)
                out_image[y] = np.squeeze(np.array(model(res).to("cpu")))
        out_image *= 255
        out = np.where(out_image >= 255, 255, out_image)
        out = out.astype(np.uint8)
        out = Image.fromarray(out)
        out.save(out_path)


def check_ssim(filtered_images, genuine_images, image_name) -> None:
    filtered_img = np.array(ImageOps.grayscale(Image.open(filtered_images / image_name)))
    genuine_img = np.array(ImageOps.grayscale(Image.open(genuine_images / image_name)))
    ssim_metric = ssim(filtered_img, genuine_img)
    print(f"{image_name}, SSIM = {ssim_metric:.3f}")


def get_dataset_name(win_size, step, path_to_csv, classification=False):
    if classification:
        datasets_list = listdir(path_to_csv / "classification")
    else:
        datasets_list = listdir(path_to_csv)
    part_of_name = f"W{win_size}_S{step}_L"
    for name in datasets_list:
        if part_of_name in name:
            if classification:
                return f"classification\{name}"
            else:
                return name
    raise Exception('Dataset absence')


def check_gmsd(filtered_images, genuine_images, image_name, rescale=True, returnMap=False):

    vref = np.array(ImageOps.grayscale(Image.open(filtered_images / image_name)))
    vcmp = np.array(ImageOps.grayscale(Image.open(genuine_images / image_name)))

    if rescale:
        scl = (255.0 / vref.max())
    else:
        scl = np.float32(1.0)

    T = 170.0
    dwn = 2
    dx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.0
    dy = dx.T

    ukrn = np.ones((2, 2)) / 4.0
    aveY1 = signal.convolve2d(scl * vref, ukrn, mode='same', boundary='symm')
    aveY2 = signal.convolve2d(scl * vcmp, ukrn, mode='same', boundary='symm')
    Y1 = aveY1[0::dwn, 0::dwn]
    Y2 = aveY2[0::dwn, 0::dwn]

    IxY1 = signal.convolve2d(Y1, dx, mode='same', boundary='symm')
    IyY1 = signal.convolve2d(Y1, dy, mode='same', boundary='symm')
    grdMap1 = np.sqrt(IxY1**2 + IyY1**2)

    IxY2 = signal.convolve2d(Y2, dx, mode='same', boundary='symm')
    IyY2 = signal.convolve2d(Y2, dy, mode='same', boundary='symm')
    grdMap2 = np.sqrt(IxY2**2 + IyY2**2)

    quality_map = (2*grdMap1*grdMap2 + T) / (grdMap1**2 + grdMap2**2 + T)
    score = np.std(quality_map)

    # if returnMap:
    #     return (score, quality_map)
    # else:
    #     return score
    print(f"{image_name}, GMSD = {score:.3f}")
    