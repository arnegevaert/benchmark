from typing import Tuple, Optional
import h5py
from numpy import typing as npt


class HDF5DatasetWriter:
    def __init__(self, path: str, shape: Tuple = None):
        self.path: str = path
        self.file: Optional[h5py.File] = None
        self.length: int = shape[0]
        self.sample_shape: Tuple = shape[1:]
        self.head = 0
        self.clear()

    def clear(self):
        if self.file is not None:
            self.file.close()
        with h5py.File(self.path, "w") as fp:
            fp.create_dataset("samples", shape=(self.length, *self.sample_shape))
            fp.create_dataset("labels", shape=(self.length,))
            self.head = 0

    def write(self, samples: npt.NDArray, labels: npt.NDArray):
        if self.length - self.head < samples.shape[0]:
            raise ValueError("Data size is exceeding pre-specified length")
        if samples.shape[0] != labels.shape[0]:
            raise ValueError("Number of samples and number of labels do not match.")
        if samples.shape[1:] != self.sample_shape:
            raise ValueError("Invalid sample shape.")
        if self.file is None:
            self.file = h5py.File(self.path, "a")

        self.file["samples"][self.head:self.head + samples.shape[0], ...] = samples
        self.file["labels"][self.head:self.head + labels.shape[0]] = labels
        self.head += samples.shape[0]

    def __del__(self):
        if self.file is not None:
            self.file.close()
