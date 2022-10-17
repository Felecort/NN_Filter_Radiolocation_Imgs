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

    img = np.array(add_borders(img, win_size)) / 255
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


def check_ssim(filtered_images, genuine_images) -> None:
    filtered_imgs_list = listdir(filtered_images)
    for image_name in filtered_imgs_list:
        filtered_img = np.array(Image.open(f"{filtered_images}\{image_name}"))
        genuine_img = np.array(Image.open(f"{genuine_images}\{image_name}"))
        ssim_metric = ssim(filtered_img, genuine_img)
        print(f"{image_name}, SSIM = {ssim_metric:.2f}")

def get_dataset_name(win_size, step, path_to_csv):
    datasets_list = listdir(path_to_csv)
    part_of_name = f"W{win_size}_S{step}_L"
    for name in datasets_list:
        if part_of_name in name:
            return name
    raise Exception('Dataset absence')
    