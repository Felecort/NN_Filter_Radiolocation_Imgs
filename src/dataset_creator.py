import numpy as np
import csv
from random import randint
from os import listdir
from PIL import ImageOps, Image
from image_processing import *
from check_validity_of_values import *
from assistive_funcs import *


def generate_csv(*, win_size, dump_to_file=1000, step=1,
                 img_path=r"..\data\images",
                 datasets_path=r"..\data\csv_files",
                 noise_imgs_path=r"..\data\FC_imgs_with_noise",
                 dataset_name=None,
                 force_create_dataset=False,
                 classification=False) -> None:
    """
    This function create dataset using certan
    window and step on the images with borders.
    """
    check_valid_win_size(win_size)
    need_to_add_name = check_dataset_name(dataset_name)

    # Define constants
    counter = 0
    half_win_size = win_size // 2
    win_square = win_size ** 2
    total_errors = 0

    list_of_img_names = sorted(listdir(img_path))
    # Array that contains rows with flatten cropped images
    dumped_data = np.empty((dump_to_file, win_square + 1), dtype=float)

    # Assistive function to show pased time
    start_time = delta_time()

    # Load the images and convert to grayscale
    imgs_list = load_images(img_path, list_of_img_names)

    # Get shuffled indexes of imgs, rows, cols and the number of samples
    m_shuffled_idxs, m_keys_list, total_length = get_shuffled_idxs(imgs_list=imgs_list, step=step)
    
    if need_to_add_name:
        dataset_name = f"W{win_size}_S{step}_L{total_length}.csv"
    if not force_create_dataset:
        check_existing_datasets(dataset_name, datasets_path)

    # Adding borders for each image
    src_images = [np.array(add_borders(img, half_win_size))
                        for img in imgs_list]
    
    parsed_imgs_list = [np.around(add_noise(img)) for img in src_images]

    for name, noised_image in zip(list_of_img_names, parsed_imgs_list):
        img = Image.fromarray(noised_image).convert("L")
        img = img.crop((half_win_size, half_win_size,
                        img.size[0] - half_win_size, img.size[1] - half_win_size))
        img.save(f"{noise_imgs_path}\\{name}")
        img = np.array(img)

    del imgs_list

    print('=' * 61, f"\nBorders were added, indexes were created. Passed time = {start_time():.2f}s")

    if classification:
        path_to_dataset = f"{datasets_path}\classification\{dataset_name}"
    else:
        path_to_dataset = f"{datasets_path}\{dataset_name}"
    with open(path_to_dataset, "w", newline='') as f:

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
            target_img = src_images[m_chosen_img_idx]
            # Get image 'window'
            cropped_img = main_img[m_row:m_row+win_size, m_column:m_column+win_size]
            cropped_src_img = target_img[m_row:m_row+win_size, m_column:m_column+win_size]

            index_in_total_data = counter % dump_to_file
            # Define target value and the noised data
            try:
                target = cropped_src_img[half_win_size, half_win_size]
                data = cropped_img.flatten()

                # Adding data for a special list in certan row
                dumped_data[index_in_total_data][:win_square] = data
                dumped_data[index_in_total_data][-1] = target
            except (IndexError, ValueError) as ierr:
                total_errors += 1
                # print(ierr)
                pass

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

    print(f"""\r=============================================================
              \rDataset created.               
              \rTotal spent time = {start_time():.2f}s
              \rTotal samples = {total_length}
              \rDataset name '{dataset_name}'
              \rErrors: {total_errors}""")



def generate_unshuffled_csv(*, win_size, dump_to_file=1000, step=1,
                 img_path=r"..\data\images",
                 datasets_path=r"..\data\csv_files",
                 noise_imgs_path=r"..\data\FC_imgs_with_noise",
                 dataset_name=None,
                 force_create_dataset=False,
                 classification=False) -> None:
    check_valid_win_size(win_size)
    need_to_add_name = check_dataset_name(dataset_name)

    # Define constants
    counter = 0
    total_samples = 0
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
        
    dumped_data = np.empty((dump_to_file, win_square + 1), dtype=float)
    with open(path_to_dataset, "w", newline='') as f:
        # Set params to csv writter
        csv.register_dialect('datasets_creator', delimiter=',', quoting=csv.QUOTE_NONE, skipinitialspace=False)
        writer_obj = csv.writer(f, dialect="datasets_creator")
        
        for image_counter, (img, name) in enumerate(zip(imgs_list, list_of_img_names), start=1):
            original_img_b = np.array(add_borders(img, half_win_size))
            # noised_img_b = np.around(add_noise(original_img_b))
            noised_img_b = add_noise(original_img_b)
            
            img = Image.fromarray(noised_img_b).convert("L")
            img = img.crop((half_win_size, half_win_size, img.size[0] - half_win_size, img.size[1] - half_win_size))
            img.save(f"{noise_imgs_path}\\{name}")
            
            height, width = noised_img_b.shape
            
            original_img_b = original_img_b / 255
            noised_img_b = noised_img_b / 255
            
            for y in range(0, height - win_size, step):
                for x in range(0, width - win_size, step):
                    data_slice = noised_img_b[y:y+win_size, x:x+win_size].flatten()
                    target = original_img_b[y + half_win_size, x + half_win_size]
                    dumped_data[counter][:win_square] = data_slice
                    dumped_data[counter][-1] = target
                    
                    counter += 1
                    if counter % dump_to_file == 0:
                        total_samples += counter
                        counter = 0
                        print(f"\rimg: {image_counter}/{total_images}", end="")
                        
                        writer_obj.writerows(dumped_data)
                        # dumped_data = np.empty((dump_to_file, win_square + 1), dtype=int)
        writer_obj.writerows(dumped_data[:counter])
        total_samples += counter
        print(f"\nSamples: {total_samples}")


if __name__ == "__main__":
    generate_unshuffled_csv(win_size=7, dump_to_file=1000, step=5,
                 img_path=r"D:\Projects\PythonProjects\NIR\data\large_data\train_images",
                 datasets_path=r"D:\Projects\PythonProjects\NIR\data\large_data\csv_files",
                 noise_imgs_path=r"D:\Projects\PythonProjects\NIR\data\large_data\train_images_noised",
                 dataset_name=None,
                 force_create_dataset=1,
                 classification=False)