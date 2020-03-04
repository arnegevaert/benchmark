from typing import Iterable, Tuple


class Dataset:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.mask_value = 0.

    def get_train_data(self) -> Iterable:
        raise NotImplementedError

    def get_test_data(self) -> Iterable:
        raise NotImplementedError

    def get_sample_shape(self) -> Tuple:
        raise NotImplementedError

    def get_batch_size(self) -> int:
        return self.batch_size
