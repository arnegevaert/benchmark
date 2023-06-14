from typing import Tuple, Optional
import h5py
from numpy import typing as npt
import numpy as np


class HDF5DatasetWriter:
    def __init__(self, path: str, num_samples: int, sample_shape: Tuple = None):
        self.path: str = path
        self.file: Optional[h5py.File] = None
        self.num_samples: int = num_samples
        self.sample_shape: Tuple = sample_shape
        self.head = 0
        self.clear()

    def clear(self):
        if self.file is not None:
            self.file.close()
        with h5py.File(self.path, "x") as fp:
            fp.create_dataset("samples", shape=(self.num_samples, *self.sample_shape), dtype=np.float32)
            fp.create_dataset("labels", shape=(self.num_samples,), dtype=np.int64)
            self.head = 0

    def write(self, samples: npt.NDArray, labels: npt.NDArray):
        if self.num_samples - self.head < samples.shape[0]:
            raise ValueError("Data size exceeds pre-specified length")
        if samples.shape[0] != labels.shape[0]:
            raise ValueError("Number of samples and number of labels do not match.")
        if samples.shape[1:] != self.sample_shape:
            raise ValueError("Invalid sample shape.")
        if self.file is None:
            self.file = h5py.File(self.path, "a")

        self.file["samples"][self.head:self.head + samples.shape[0], ...] = samples.astype(np.float32)
        self.file["labels"][self.head:self.head + labels.shape[0]] = labels.astype(np.int64)
        self.head += samples.shape[0]

    def __del__(self):
        if self.file is not None:
            self.file.close()
