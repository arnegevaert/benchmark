from typing import Tuple
from .._typing import _check_is_dataset
import h5py
from numpy import typing as npt


class AttributionsDatasetWriter:
    def __init__(self, path: str, num_samples: int):
        self.path = path
        self.num_samples: int = num_samples
        self.sample_shape: Tuple[int, ...] | None = None
        self.file = h5py.File(self.path, "w")
        self.file.attrs["num_samples"] = self.num_samples

    def write(
        self, indices: npt.NDArray, attributions: npt.NDArray, method_name: str
    ):
        if self.sample_shape is None:
            self.sample_shape = attributions.shape[1:]
            self.file.attrs["sample_shape"] = self.sample_shape
        if attributions.shape[1:] != self.sample_shape:
            raise ValueError(
                f"Invalid sample shape. Expected: {self.sample_shape}, "
                f"got: {attributions.shape[1:]}"
            )
        if method_name not in self.file.keys():
            dataset = self.file.create_dataset(
                method_name, shape=(self.num_samples, *self.sample_shape)
            )
        else:
            dataset = _check_is_dataset(self.file[method_name])
        dataset[indices, ...] = attributions
