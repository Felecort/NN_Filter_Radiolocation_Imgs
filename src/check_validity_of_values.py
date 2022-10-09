from os import listdir


def assign_name_to_dataset(dataset_name, win_size) -> str:
    """
    Return the valid dataset_name
    """
    if dataset_name is None:
        dataset_name = f"data_win{win_size}.csv"
    else:
        assert isinstance(dataset_name, str), "Dataset name shuold be str"
        assert len(dataset_name) > 0, "Name shouldn't be empty line"
        dataset_name = dataset_name.replace(" ", "_")
        if dataset_name[-4:] != ".csv":
            dataset_name += ".csv"
    return dataset_name


def check_existing_datasets(dataset_name, datasets_path) -> None:
    """
    Checks if the dataset exists.
    If True new dataset won't be created
    """
    datasets = listdir(datasets_path)
    assert dataset_name not in datasets,\
        f"Dataset '{dataset_name}' already exists, change the window size"
