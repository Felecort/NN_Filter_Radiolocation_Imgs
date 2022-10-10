import numpy as np
import csv
from random import randint
from os import listdir
from image_processing import *
from check_validity_of_values import *
from assistive_funcs import *


def generate_csv(*, win_size, dump_to_file=1000, step=1,
                 img_path=r"..\datasets\images",
                 datasets_path=r"..\datasets\csv_files",
                 dataset_name=None,
                 force_create_dataset=False) -> None:
    """
    This function create dataset using certan
    window and step on the images with borders.
    """
    #############################################################
    # Checking valid name, win_size and existing dataset
    check_valid_win_size(win_size)
    dataset_name = assign_name_to_dataset(dataset_name, win_size, step)
    if not force_create_dataset:
        check_existing_datasets(dataset_name, datasets_path)
    #############################################################

    # Define constants
    counter = 0
    half_win_size = win_size // 2
    win_square = win_size ** 2

    list_of_img_names = listdir(img_path)

    # Array that contains rows with flatten cropped images
    dumped_data = np.empty((dump_to_file, win_square + 1), dtype=float)

    # Assistive function to show pased time
    start_time = delta_time()

    # Load the images and convert to grayscale
    imgs_list = load_images(img_path, list_of_img_names)

    # Get shuffled indexes of imgs, rows, cols and the number of samples
    m_shuffled_idxs, m_keys_list, total_length = get_shuffled_idxs(imgs_list=imgs_list, step=step)

    # Adding borders for each image
    parsed_imgs_list = [np.array(add_borders(img, half_win_size)) / 255
                        for img in imgs_list]
    del imgs_list

    print(f"\nBorders were added, indexes were created. Passed time = {start_time():.2f}s")

    with open(f"{datasets_path}\{dataset_name}", "w", newline='') as f:

        # Set params to csv writter
        csv.register_dialect('datasets_creator', delimiter=',', quoting=csv.QUOTE_NONE, skipinitialspace=False)
        writer_obj = csv.writer(f, dialect="datasets_creator")

        while m_shuffled_idxs:
            counter += 1

            """ This section choose random image, row and column """
            """ The Devil will break his leg here """
            ############################################################
            # Random choose image, m_row(key) and val(m_column)
            # Get an index if random image
            m_chosen_img_idx = randint(0, len(m_shuffled_idxs) - 1)
            # Get y indexes-m_column, the keys in a dict
            m_chosen_keys = m_keys_list[m_chosen_img_idx]
            # Get random key-m_row
            m_row_idx = randint(0, len(m_chosen_keys) - 1)
            m_row = m_keys_list[m_chosen_img_idx][m_row_idx]
            # Get last value. Values have been just shuffled
            m_column = m_shuffled_idxs[m_chosen_img_idx][m_row].pop()
            ############################################################

            # Choose image
            main_img = parsed_imgs_list[m_chosen_img_idx]
            # Get image 'window'
            cropped_img = main_img[m_row:m_row+win_size, m_column:m_column+win_size]

            # Define target value and the noised data
            target = cropped_img[half_win_size, half_win_size]
            data = add_noise(cropped_img).flatten()

            # Adding data for a special list in certan row
            index_in_total_data = counter % dump_to_file
            dumped_data[index_in_total_data][:win_square] = data
            dumped_data[index_in_total_data][-1] = target

            # Dump to file
            if index_in_total_data == 0:
                writer_obj.writerows(dumped_data)
                print(f"\rTime left = {start_time():.2f}s, {(counter / total_length) * 100:.2f}%", end="")

            """ When no indexes in the line or no columns in image, removed keys, img"""
            """ The Devil will break his leg here """
            ############################################################
            if len(m_shuffled_idxs[m_chosen_img_idx][m_row]) == 0:
                m_shuffled_idxs[m_chosen_img_idx].pop(m_row)
                m_keys_list[m_chosen_img_idx].pop(m_row_idx)
                if len(m_shuffled_idxs[m_chosen_img_idx]) == 0:
                    m_shuffled_idxs.pop(m_chosen_img_idx)
                    m_keys_list.pop(m_chosen_img_idx)
            ############################################################
        # Dump to file rows, that weren't be added in the end
        writer_obj.writerows(dumped_data[:index_in_total_data])

    print(f"""\rDataset created.               
              \rTotal spent time = {start_time():.2f}s
              \rTotal samples = {total_length}
              \rDataset name '{dataset_name}'""")
