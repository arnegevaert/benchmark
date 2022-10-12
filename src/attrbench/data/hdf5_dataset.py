from torch.utils.data import Dataset
from typing import Optional
import h5py


class HDF5Dataset(Dataset):
    """
    File
    - samples: [num_samples, *sample_shape]
    - labels: [num_samples]
    """
    def __init__(self, path: str):
        self.path = path
        self.file: Optional[h5py.File] = None

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, "r")
        return self.file["samples"][index], self.file["labels"][index]

    def __len__(self):
        if self.file is None:
            self.file = h5py.File(self.path, "r")
        return len(self.file["samples"])

    def __del__(self):
        if self.file is not None:
            self.file.close()
