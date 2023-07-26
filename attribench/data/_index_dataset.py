from collections.abc import Sized
from torch.utils.data import Dataset


class IndexDataset(Dataset):
    """Wraps a dataset to return the index of the sample as well.
    Used internally to keep track of the index of a sample in a dataset.
    """

    def __init__(self, dataset: Dataset):
        """
        Parameters
        ----------
        dataset : Dataset
            The PyTorch Dataset to wrap.
        """
        super().__init__()
        self.dataset: Dataset = dataset

    def __len__(self):
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        else:
            raise ValueError("Base dataset has no __len__")

    def __getitem__(self, item):
        data, target = self.dataset[item]
        return item, data, target
