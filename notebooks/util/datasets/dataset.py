from typing import Tuple, Iterable


class Dataset:
    def __init__(self, batch_size, sample_shape):
        self.batch_size = batch_size
        self.sample_shape = sample_shape

    def get_sample_shape(self) -> Tuple:
        raise self.sample_shape

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_train_data(self) -> Iterable:
        raise NotImplementedError

    def get_test_data(self) -> Iterable:
        raise NotImplementedError
