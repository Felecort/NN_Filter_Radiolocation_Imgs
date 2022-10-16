import pandas as pd
from pathlib import Path
from math import floor


# Paths
main_data_path = Path("../data")
scv_data = main_data_path / "csv_files" # scv_data
img_path = main_data_path / "images"


class CustomDataLoader:
    def __init__(self, *, scv_data, dataset_name, batch_size, train_size, is_train):
        self.scv_data = scv_data
        self.dataset_name = dataset_name
        self.main_path = scv_data / dataset_name

        self.batch_size = batch_size
        self.train_size = train_size

        self.test_batches, self.real_test_samples = self.get_test_amount()
        self.skip_rows = self.real_test_samples

        self.is_train = is_train
        self.counter = 0

        if self.is_train:
            self.data = pd.read_csv(self.main_path, chunksize=self.batch_size,
                                    header=None, index_col=None, iterator=True)
        else:
            self.data = pd.read_csv(self.main_path, chunksize=self.batch_size,
                                    header=None, index_col=None, iterator=True,
                                    skiprows=self.skip_rows)

    def get_len_data(self) -> int:
        idx_start = self.dataset_name.find("L") + 1
        idx_finish = self.dataset_name.find(".")
        length = int(self.dataset_name[idx_start:idx_finish])
        return length

    def get_test_amount(self) -> tuple:
        length = self.get_len_data()
        test_smaples = int(length * self.train_size)
        test_batches = floor(test_smaples / self.batch_size)
        real_test_samples = test_batches * self.batch_size
        print(f"{length = }, {test_batches = }, {real_test_samples = }")
        return test_batches, real_test_samples

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_train:
            while self.counter < self.test_batches:
                self.counter += 1
                return self.data.get_chunk()
            raise StopIteration
        else:
            return self.data.get_chunk()


def get_train_test(*, scv_data, dataset_name, batch_size, train_size):
    train = CustomDataLoader(scv_data=scv_data,
                         dataset_name=dataset_name,
                         batch_size=batch_size,
                         train_size=train_size,
                         is_train=True)
    test = CustomDataLoader(scv_data=scv_data,
                        dataset_name=dataset_name,
                        batch_size=batch_size,
                        train_size=train_size,
                        is_train=False)
    return train, test