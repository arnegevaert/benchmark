from collections.abc import Sized
from torch.utils.data import Dataset


class IndexDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset: Dataset = dataset

    def __len__(self):
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        else:
            raise ValueError("Base dataset has no __len__")

    def __getitem__(self, item):
        data, target = self.dataset[item]
        return item, data, target
