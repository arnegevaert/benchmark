from torch.utils.data import Dataset, Sampler
from torch.utils.data.sampler import T_co
from typing import Iterator


class ParallelEvalSampler(Sampler):
    def __init__(self, dataset: Dataset, world_size: int, rank: int):
        """
        DistributedEvalSampler is an alternative to DistributedSampler that
        does not add extra samples to make the total count evenly divisible.
        It also does not shuffle the data.
        Credit: https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py
        """
        super().__init__(None)
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank

        ds_size = len(dataset)
        self.indices = list(range(ds_size))[self.rank:ds_size:self.world_size]

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.indices)


class IndexDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data, target = self.dataset[item]
        return item, data, target
