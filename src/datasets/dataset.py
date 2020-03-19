from typing import Iterable, Tuple
import torch


class Dataset:
    def __init__(self, batch_size, transforms):
        self.batch_size = batch_size
        self.mask_value = 0.
        self.transforms = transforms

    def get_train_data(self) -> Iterable:
        raise NotImplementedError

    def get_test_data(self) -> Iterable:
        raise NotImplementedError

    def get_sample_shape(self) -> Tuple:
        raise NotImplementedError

    def get_batch_size(self) -> int:
        return self.batch_size

    def transform_batch(self, batch):
        result = []
        for i in range(batch.shape[0]):
            transformed = batch[i]
            for t in self.transforms:
                transformed = t(transformed)
            result.append(transformed)
        return torch.cat(result, dim=0)
