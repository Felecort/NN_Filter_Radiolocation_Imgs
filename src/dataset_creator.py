import pandas as pd
import numpy as np
from os import listdir
from PIL import Image, ImageOps
from tqdm import tqdm
from data_preprocessing import *


# Define global constants
datasets_path = r"..\datasets\csv_files"
img_path = r"..\datasets\images"


def generate_csv(*, win_size,
                 dump_to_file,
                 img_path=img_path,
                 datasets_path=datasets_path,
                 dataset_name=None):
    """
    This function create dataset using certan
    window on images with borders.
    """

    #############################################################
    # Checking valid name, win_size and existing dataset
    assert (win_size % 2) == 1, "The win_size should be odd"

    dataset_name = assign_name_to_dataset(dataset_name, win_size)

    check_existing_datasets(dataset_name, datasets_path)
    #############################################################

    # Define constants
    counter = 0
    win_size_square = win_size * win_size
    data_arr = np.empty((dump_to_file, win_size_square + 1), dtype=float)
    imgs_list = listdir(img_path)
    
    """ Test dataset shuffle """
    # shuffled_idxs = get_total_length(img_path, imgs_list)
    # print(shuffled_idxs[0][:20])
    # print(shuffled_idxs[2][:20])
    # return
    """ Test dataset shuffle """
    
    for file_name in tqdm(imgs_list):

        # Load and convert image
        path = f"{img_path}\{file_name}"
        img = ImageOps.grayscale(Image.open(path))

        # Creatre border
        half_win_size = win_size // 2
        img_with_borders = add_borders(img, half_win_size,
                                       img.size[0], img.size[1])

        # Normalization
        norm_img_with_borders = np.array(img_with_borders) / 255

        for y in range(half_win_size, half_win_size + img.size[1]):
            for x in range(half_win_size, half_win_size + img.size[0]):

                # Define margins
                left = x - half_win_size
                top = y - half_win_size
                right = x + half_win_size + 1
                bottom = y + half_win_size + 1

                # Get cropped image
                cropped_img = norm_img_with_borders[top:bottom, left:right]

                target_pixel = cropped_img[half_win_size, half_win_size]

                # Add noise and flatten
                noised_img = add_noise(cropped_img, win_size).flatten()

                data_arr[counter, :win_size_square] = noised_img
                data_arr[counter, win_size_square] = target_pixel

                counter += 1
                if (counter % dump_to_file) == 0:
                    df = pd.DataFrame(data_arr)

                    # 0...n, target
                    df.to_csv(f"{datasets_path}\{dataset_name}",
                              sep=",", mode='a', index=False, header=False)
                    counter = 0

    df = pd.DataFrame(data_arr[:counter])
    # [0, ..., n, target]
    df.to_csv(f"{datasets_path}\data_win{win_size}.csv",
              sep=",", mode='a', index=False, header=False)
