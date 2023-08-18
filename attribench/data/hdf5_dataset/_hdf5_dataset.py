from torch.utils.data import Dataset
from .._typing import _check_is_dataset
from typing import Tuple
import h5py


class HDF5Dataset(Dataset):
    """
    Dataset stored in a HDF5 file.

    The HDF5 file must contain the following datasets:

    - ``samples: [num_samples, *sample_shape]``
    - ``labels: [num_samples]``
    """

    def __init__(self, path: str):
        """
        Parameters
        ----------
        path : str
            Path to the HDF5 file.
        """
        self.path = path
        self.file: h5py.File | None = None
        self._sample_shape: Tuple | None = None

    @property
    def sample_shape(self):
        if self.file is None:
            with h5py.File(self.path, "r") as fp:
                return _check_is_dataset(fp["samples"]).shape[1:]
        return _check_is_dataset(self.file["samples"]).shape[1:]

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, "r")
        return (
            _check_is_dataset(self.file["samples"])[index],
            _check_is_dataset(self.file["labels"])[index],
        )

    def __len__(self):
        if self.file is None:
            with h5py.File(self.path, "r") as fp:
                return len(_check_is_dataset(fp["samples"]))
        return len(_check_is_dataset(self.file["samples"]))