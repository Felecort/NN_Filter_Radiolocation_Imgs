from random import choice, randint
import pandas as pd
import numpy as np
from os import listdir
from PIL import Image, ImageOps
from tqdm import tqdm
from image_processing import *
from check_validity_of_values import *
from pprint import pprint


# Define global constants
datasets_path = r"..\datasets\csv_files"
img_path = r"..\datasets\images"


def generate_csv(*, win_size: int,
                 dump_to_file: int,
                 step:int = 1,
                 img_path=img_path,
                 datasets_path=datasets_path,
                 dataset_name=None) -> None:
    """
    This function create dataset using certan
    window on images with borders.
    """
    #############################################################
    # Checking valid name, win_size and existing dataset
    assert ((win_size % 2) == 1) and (win_size > 0), "The win_size should be odd positive"
    dataset_name = assign_name_to_dataset(dataset_name, win_size)
    check_existing_datasets(dataset_name, datasets_path)
    #############################################################

    # Define constants
    counter = 0
    # win_size_square = win_size * win_size
    # half_win_size = win_size // 2
    # data_arr = np.empty((dump_to_file, win_size_square + 1), dtype=float)
    list_of_img_names = listdir(img_path)
    create_dataset = True
    
    load_images(img_path, list_of_img_names)
    return
    
    shuffled_idxs, keys_list = get_shuffled_idxs(img_path=img_path, list_of_img_names=list_of_img_names, step=step)
    
    
    while create_dataset:
        # Random choose image, row(key) and val(column)
        
        # Get an index if random image 
        chosen_img_idx = randint(0, len(shuffled_idxs) - 1)
        
        # Get image by a random index
        chosen_img = shuffled_idxs[chosen_img_idx]
        
        # Get y indexes-column, the keys in a dict 
        chosen_keys = keys_list[chosen_img_idx]
        
        # Get random key-row 
        row = choice(chosen_keys)
        
        # Get row by random index(key) in a dict
        values = chosen_img[row]
        
        # Get last value. Values have been just shuffled
        column = values.pop()
        
        
        
        return



if __name__ == "__main__":
    generate_csv(win_size=7, dump_to_file=1000, step=100,
                 img_path=r"D:\Projects\PythonProjects\NIR\datasets\images",
                 datasets_path=r"D:\Projects\PythonProjects\NIR\datasets\csv_files")
    
    # for file_name in tqdm(list_of_img_names):

    #     # Load and convert image
    #     img = ImageOps.grayscale(Image.open(f"{img_path}\{file_name}"))

    #     # Creatre border
    #     img_with_borders = add_borders(img, half_win_size,
    #                                    img.size[0], img.size[1])

    #     # Normalization
    #     norm_img_with_borders = np.array(img_with_borders) / 255

    #     for y in range(half_win_size, half_win_size + img.size[1]):
    #         for x in range(half_win_size, half_win_size + img.size[0]):

    #             # Define margins
    #             left = x - half_win_size
    #             top = y - half_win_size
    #             right = x + half_win_size + 1
    #             bottom = y + half_win_size + 1

    #             # Get cropped image
    #             cropped_img = norm_img_with_borders[top:bottom, left:right]

    #             target_pixel = cropped_img[half_win_size, half_win_size]

    #             # Add noise and flatten
    #             noised_img = add_noise(cropped_img, win_size).flatten()

    #             data_arr[counter, :win_size_square] = noised_img
    #             data_arr[counter, win_size_square] = target_pixel

    #             counter += 1
    #             if (counter % dump_to_file) == 0:
    #                 df = pd.DataFrame(data_arr)

    #                 # 0...n, target
    #                 df.to_csv(f"{datasets_path}\{dataset_name}",
    #                           sep=",", mode='a', index=False, header=False)
    #                 counter = 0

    # df = pd.DataFrame(data_arr[:counter])
    # # [0, ..., n, target]
    # df.to_csv(f"{datasets_path}\data_win{win_size}.csv",
    #           sep=",", mode='a', index=False, header=False)
