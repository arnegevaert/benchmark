from torch.utils.data import Dataset
from typing import Tuple
import h5py


class HDF5Dataset(Dataset):
    """
    File
    - samples: [num_samples, *sample_shape]
    - labels: [num_samples]
    """
    def __init__(self, path: str):
        self.path = path
        self.file: h5py.File | None = None
        self._sample_shape: Tuple | None = None

    @property
    def sample_shape(self):
        if self.file is None:
            with h5py.File(self.path, "r") as fp:
                return fp["samples"].shape[1:]
        return self.file["samples"].shape[1:]

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, "r")
        return self.file["samples"][index], self.file["labels"][index]

    def __len__(self):
        if self.file is None:
            with h5py.File(self.path, "r") as fp:
                return len(fp["samples"])
        return len(self.file["samples"])

    def __del__(self):
        if self.file is not None:
            self.file.close()
