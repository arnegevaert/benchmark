from torch.utils.data import Dataset


class IndexDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data, target = self.dataset[item]
        return item, data, target
