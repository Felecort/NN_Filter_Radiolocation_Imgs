from random import randint
import numpy as np
from os import listdir
from image_processing import *
from check_validity_of_values import *
from assistive_funcs import *
import csv

# Define global constants
datasets_path = r"..\datasets\csv_files"
img_path = r"..\datasets\images"



def generate_csv(*, win_size, dump_to_file=1000, step=1,
                 img_path=img_path,
                 datasets_path=datasets_path,
                 dataset_name=None, 
                 force_create_dataset=False) -> None:
    """
    This function create dataset using certan
    window on images with borders.
    """
    #############################################################
    # Checking valid name, win_size and existing dataset
    check_valid_win_size(win_size)
    dataset_name = assign_name_to_dataset(dataset_name, win_size, step)
    if not force_create_dataset: check_existing_datasets(dataset_name, datasets_path)
    #############################################################

    # Define constants
    counter = 0
    half_win_size = win_size // 2
    win_square = win_size ** 2
    
    list_of_img_names = listdir(img_path)
    
    total_data = np.empty((dump_to_file, win_square + 1), dtype=float)
    
    start_time = delta_time()

    # Load the images and convert to grayscale
    imgs_list = load_images(img_path, list_of_img_names)

    m_shuffled_idxs, m_keys_list, total_length = get_shuffled_idxs(imgs_list=imgs_list, step=step)

    # Adding a border for each image
    parsed_imgs_list = [np.array(add_borders(img, half_win_size)) / 255 for img in imgs_list]
    
    print(f"\nBorders were added, indexes created. passed time = {start_time():.2f}")
        
    del imgs_list
    
    with open(f"{datasets_path}\{dataset_name}", "w", newline='') as f:
        
        csv.register_dialect('datasets_creator',
                            delimiter=',',
                            quoting=csv.QUOTE_NONE,
                            skipinitialspace=False)
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
            
            main_img = parsed_imgs_list[m_chosen_img_idx]
            cropped_img = main_img[m_row:m_row+win_size, m_column:m_column+win_size]
            
            target = cropped_img[half_win_size, half_win_size]
            data = add_noise(cropped_img).flatten()
            
            index_in_total_data = counter % dump_to_file
            total_data[index_in_total_data][:win_square] = data
            total_data[index_in_total_data][-1] = target
            
            if index_in_total_data == 0:
                
                writer_obj.writerows(total_data)
                print(f"\rTime left = {start_time():.2f}, {(counter / total_length) * 100:.2f}%", end="")
            
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
        writer_obj.writerows(total_data[:index_in_total_data])
        
    print(f"""\nDataset created.
              \rTotal spent time = {start_time():.2f}
              \rTotal samples = {total_length}
              \rDataset name '{dataset_name}'""")

if __name__ == "__main__":
    generate_csv(win_size=11, dump_to_file=1000, step=1,
                 img_path=r"D:\Projects\PythonProjects\NIR\datasets\images",
                 datasets_path=r"D:\Projects\PythonProjects\NIR\datasets\csv_files",
                 force_create_dataset=True)
