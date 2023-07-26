from torch.utils.data import Dataset, Sampler
from torch.utils.data.sampler import T_co
from typing import Iterator
import random


class DistributedSampler(Sampler):
    def __init__(self, dataset: Dataset, world_size: int, rank: int, shuffle=True):
        """
        DistributedSampler is an alternative to the PyTorch DistributedSampler that
        does not add extra samples to make the total count evenly divisible.
        Credit: https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py
        """
        super().__init__(None)
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank

        ds_size = len(dataset)
        self.indices = list(range(ds_size))[self.rank:ds_size:self.world_size]
        if shuffle:
            random.shuffle(self.indices)

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.indices)
