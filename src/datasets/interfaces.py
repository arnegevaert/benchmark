from typing import Tuple
from torch.utils.data import DataLoader


# Interface to represent a general dataset.
class Dataset:
    def __init__(self, batch_size, sample_shape):
        self.batch_size = batch_size
        self.sample_shape = sample_shape

    def get_sample_shape(self) -> Tuple:
        raise self.sample_shape

    def get_batch_size(self) -> int:
        return self.batch_size


# Interface to represent a dataset with a separate train and test set.
class TrainableDataset(Dataset):
    def get_sample_shape(self) -> Tuple:
        raise NotImplementedError

    def get_train_loader(self) -> DataLoader:
        raise NotImplementedError

    def get_test_loader(self) -> DataLoader:
        raise NotImplementedError
