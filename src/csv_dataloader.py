import pandas as pd
from pathlib import Path
from math import floor
from torch import Tensor
from numpy import newaxis
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
# Paths
main_data_path = Path("../data")
scv_folder = main_data_path / "csv_files"  # scv_folder
img_path = main_data_path / "images"


class _CustomBigDataLoader:
    def __init__(self, *, scv_folder, dataset_name, batch_size, train_size, is_train):
        self.scv_folder = scv_folder
        self.dataset_name = dataset_name
        self.main_path = scv_folder / dataset_name

        self.batch_size = batch_size
        self.train_size = train_size

        self.test_batches, self.real_test_samples = self._get_test_amount()
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

    def _get_len_data(self) -> int:
        idx_start = self.dataset_name.find("L") + 1
        idx_finish = self.dataset_name.find(".")
        length = int(self.dataset_name[idx_start:idx_finish])
        return length

    def _get_test_amount(self) -> tuple:
        length = self._get_len_data()
        test_smaples = int(length * self.train_size)
        test_batches = floor(test_smaples / self.batch_size)
        real_test_samples = test_batches * self.batch_size
        return test_batches, real_test_samples

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_train:
            if self.counter < self.test_batches:
                self.counter += 1
                raw_chunk = self.data.get_chunk()
                x, y = self._prepare_chunk(raw_chunk)
                return x, y
            raise StopIteration
        else:
            raw_chunk = self.data.get_chunk()
            x, y = self._prepare_chunk(raw_chunk)
            return x, y

    def _prepare_chunk(self, raw_chunk):
        x = raw_chunk.drop(columns=raw_chunk.shape[1] - 1)
        y = raw_chunk[raw_chunk.shape[1] - 1]

        x = Tensor(x.to_numpy()).float()
        y = Tensor(y.to_numpy()[:, newaxis]).float()
        return x, y


def get_train_test_big_data(*, scv_folder, dataset_name, batch_size, train_size):
    train = _CustomBigDataLoader(scv_folder=scv_folder,
                              dataset_name=dataset_name,
                              batch_size=batch_size,
                              train_size=train_size,
                              is_train=True)
    test = _CustomBigDataLoader(scv_folder=scv_folder,
                             dataset_name=dataset_name,
                             batch_size=batch_size,
                             train_size=train_size,
                             is_train=False)
    return train, test


class _CustomSmallDataLoader(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)


def get_train_test_small_data(*, scv_folder, dataset_name, batch_size, train_size):
    main_path = scv_folder / dataset_name
    data = pd.read_csv(main_path, header=None)
    y = data[data.shape[1] - 1].to_numpy()
    x = data.iloc[:, 0:data.shape[1]-1].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1-train_size, shuffle=False)
    train_data = DataLoader(_CustomSmallDataLoader(X_train, y_train), batch_size=batch_size)
    test_data = DataLoader(_CustomSmallDataLoader(X_test, y_test), batch_size=batch_size)
    return train_data, test_data
    
    