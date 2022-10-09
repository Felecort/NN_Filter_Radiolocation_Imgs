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
                 step: int = 1,
                 img_path=img_path,
                 datasets_path=datasets_path,
                 dataset_name=None) -> None:
    """
    This function create dataset using certan
    window on images with borders.
    """
    #############################################################
    # Checking valid name, win_size and existing dataset
    check_valid_win_size(win_size)
    check_existing_datasets(dataset_name, datasets_path)
    dataset_name = assign_name_to_dataset(dataset_name, win_size)
    #############################################################

    # Define constants
    counter = 0
    half_win_size = win_size // 2
    list_of_img_names = listdir(img_path)
    # win_size_square = win_size * win_size
    # data_arr = np.empty((dump_to_file, win_size_square + 1), dtype=float)

    # Load the images and convert to grayscale
    imgs_list = load_images(img_path, list_of_img_names)

    m_shuffled_idxs, m_keys_list = get_shuffled_idxs(imgs_list=imgs_list, step=step)

    # Adding a border for each image
    parsed_imgs_list = [np.array(add_borders(img, half_win_size)) / 255 for img in imgs_list]
    del imgs_list

    while m_shuffled_idxs:
        ############################################################
        # Random choose image, m_row(key) and val(m_column)
        # Get an index if random image
        m_chosen_img_idx = randint(0, len(m_shuffled_idxs) - 1)
        # Get image by a random index
        m_chosen_img = m_shuffled_idxs[m_chosen_img_idx]
        # Get y indexes-m_column, the keys in a dict
        m_chosen_keys = m_keys_list[m_chosen_img_idx]
        # Get random key-m_row
        m_row_idx = randint(0, len(m_chosen_keys) - 1)
        m_row = m_keys_list[m_chosen_img_idx][m_row_idx]
        # Get last value. Values have been just shuffled
        m_column = m_shuffled_idxs[m_chosen_img_idx][m_row].pop()
        ############################################################
        
        main_img = parsed_imgs_list[m_chosen_img_idx]
        cropped_img = main_img[m_row:m_row+win_size, m_column:m_column+win_size]
        
        target = cropped_img[half_win_size, half_win_size]
        data = add_noise(cropped_img)

        ############################################################
        if len(m_shuffled_idxs[m_chosen_img_idx][m_row]) == 0:
            m_shuffled_idxs[m_chosen_img_idx].pop(m_row)
            m_keys_list[m_chosen_img_idx].pop(m_row_idx)
            counter += 1
        if not m_shuffled_idxs[m_chosen_img_idx]:
            m_shuffled_idxs.pop(m_chosen_img_idx)
            m_keys_list.pop(m_chosen_img_idx)
        ############################################################
    print(counter)


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
