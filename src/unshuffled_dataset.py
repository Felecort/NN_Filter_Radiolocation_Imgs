import numpy as np
import csv
from random import randint
from os import listdir
from PIL import ImageOps, Image
from image_processing import *
from check_validity_of_values import *
from assistive_funcs import *


def unshuffled_dataset(*, win_size, dump_to_file=1000, step=1,
                 img_path=r"..\data\images",
                 datasets_path=r"..\data\csv_files",
                 noise_imgs_path=r"..\data\FC_imgs_with_noise",
                 dataset_name=None,
                 force_create_dataset=False,
                 classification=False) -> None:
    check_valid_win_size(win_size)
    need_to_add_name = check_dataset_name(dataset_name)
    start_time = delta_time()

    # Define constants
    counter = 0
    half_win_size = win_size // 2
    win_square = win_size ** 2
    
    list_of_img_names = sorted(listdir(img_path))
    
    imgs_list = load_images(img_path, list_of_img_names)
    total_images = len(imgs_list)
    
    if need_to_add_name:
        dataset_name = f"W{win_size}_S{step}.csv"
    if not force_create_dataset:
        check_existing_datasets(dataset_name, datasets_path)

    if classification:
        path_to_dataset = f"{datasets_path}\classification\{dataset_name}"
    else:
        path_to_dataset = f"{datasets_path}\{dataset_name}"
        
    dumped_data = np.empty((dump_to_file, win_square + 1), dtype=int)
    with open(path_to_dataset, "w", newline='') as f:
        # Set params to csv writter
        csv.register_dialect('datasets_creator', delimiter=',', quoting=csv.QUOTE_NONE, skipinitialspace=False)
        writer_obj = csv.writer(f, dialect="datasets_creator")
        
        for image_counter, (img, name) in enumerate(zip(imgs_list, list_of_img_names), start=1):
            original_img_b = np.array(add_borders(img, half_win_size))
            noised_img_b = np.around(add_noise(original_img_b))
            
            img = Image.fromarray(noised_img_b).convert("L")
            img = img.crop((half_win_size, half_win_size, img.size[0] - half_win_size, img.size[1] - half_win_size))
            img.save(f"{noise_imgs_path}\\{name}")
            height, width = noised_img_b.shape
            for y in range(0, height - win_size, step):
                for x in range(0, width - win_size, step):
                    data_slice = noised_img_b[y:y+win_size, x:x+win_size].flatten()
                    target = original_img_b[y + half_win_size, x + half_win_size]
                    dumped_data[counter][:win_square] = data_slice
                    dumped_data[counter][-1] = target
                    
                    counter += 1
                    if counter % dump_to_file == 0:
                        writer_obj.writerows(dumped_data)
                        print(f"\rTime left = {start_time():.2f}s, {image_counter}/{total_images}", end="")
                        counter = 0
                        # dumped_data = np.empty((dump_to_file, win_square + 1), dtype=int)
        writer_obj.writerows(dumped_data[:counter])


if __name__ == "__main__":
    unshuffled_dataset(win_size=7, dump_to_file=1000, step=5,
                 img_path=r"D:\Projects\PythonProjects\NIR\data\large_data\train_images",
                 datasets_path=r"D:\Projects\PythonProjects\NIR\data\large_data\csv_files",
                 noise_imgs_path=r"D:\Projects\PythonProjects\NIR\data\large_data\train_images_noised",
                 dataset_name=None,
                 force_create_dataset=1,
                 classification=False)