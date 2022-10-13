from typing import Optional, Tuple
import h5py
from numpy import typing as npt


class AttributionsDatasetWriter:
    def __init__(self, path: str, truncate=False, num_samples: int = None,
                 sample_shape: Tuple = None):
        self.path = path
        self.num_samples: Optional[int] = num_samples
        self.sample_shape: Optional[Tuple] = sample_shape
        if truncate:
            # Wipe and regenerate file using the provided metadata
            if num_samples is None or sample_shape is None:
                raise ValueError("If truncate is set, num_samples and sample_shape are required.")
            self.file = h5py.File(self.path, "w")
            self.file.attrs["num_samples"] = self.num_samples
            self.file.attrs["sample_shape"] = self.sample_shape
            self.method_names = []
        else:
            # Derive metadata from existing file
            self.file = h5py.File(self.path, "a")
            self.num_samples = self.file.attrs["num_samples"]
            self.sample_shape = self.file.attrs["sample_shape"]
            self.method_names = list(self.file["attributions"].keys())

    def write(self, indices: npt.NDArray, attributions: npt.NDArray, method_name: str):
        if method_name not in self.method_names:
            self.file.create_dataset(method_name, shape=(self.num_samples, *self.sample_shape))
            self.method_names.append(method_name)
        self.file[method_name][indices, ...] = attributions

    def __del__(self):
        self.file.close()
