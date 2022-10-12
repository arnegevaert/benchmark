from torch.utils.data import Dataset
import h5py
from typing import Optional


class AttributionsDataset(Dataset):
    """
    File
    - samples: [num_samples, *sample_shape]
    - labels: [num_samples]
    - attributions:
        - method_1: [num_samples, *attr_shape]
        - method_2: [num_samples, *attr_shape]
        - ...
    """

    def __init__(self, path: str):
        self.path = path
        self.file: Optional[h5py.File] = None
        with h5py.File(path, "r") as fp:
            self.num_samples = fp["samples"].shape[0]
            self.attribution_methods = list(fp["attributions"].keys())

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, "r")
        method_name = self.attribution_methods[index // self.num_samples]
        sample_idx = index % self.num_samples
        return sample_idx, \
               self.file["samples"][sample_idx], \
               self.file["labels"][sample_idx], \
               self.file["attributions"][method_name][sample_idx], \
               method_name

    def __len__(self):
        if self.file is None:
            self.file = h5py.File(self.path, "r")
        return self.num_samples * len(self.attribution_methods)
