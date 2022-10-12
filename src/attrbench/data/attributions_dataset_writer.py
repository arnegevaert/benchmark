from typing import Optional, Tuple
import h5py
from numpy import typing as npt


class AttributionsDatasetWriter:
    def __init__(self, path: str, truncate=False, num_samples: int = None,
                 sample_shape: Tuple = None, attributions_shape: Tuple = None):
        self.path = path
        self.num_samples: Optional[int] = num_samples
        self.sample_shape: Optional[Tuple] = sample_shape
        self.attributions_shape: Optional[Tuple] = attributions_shape
        if truncate:
            # Wipe and regenerate file using the provided shapes
            if num_samples is None or sample_shape is None or attributions_shape is None:
                raise ValueError("If truncate is set, num_samples, sample_shape and attributions_shape are required.")
            self.file = h5py.File(self.path, "w")
            self.file.create_dataset("samples", shape=(self.num_samples, *self.sample_shape))
            self.file.create_dataset("labels", shape=(self.num_samples,))
            self.file.create_group("attributions")
            self.method_names = []
        else:
            # Derive shapes from existing file
            self.file = h5py.File(self.path, "a")
            self.num_samples = self.file["samples"].shape[0]
            self.sample_shape = self.file["samples"].shape[1:]
            self.method_names = list(self.file["attributions"].keys())
            self.attributions_shape = self.file["attributions"][self.method_names[0]].shape[1:]

    def write_samples(self, indices: npt.NDArray, samples: npt.NDArray, labels: npt.NDArray):
        self.file["samples"][indices, ...] = samples
        self.file["labels"][indices, ...] = labels

    def write_attributions(self, indices: npt.NDArray, attributions: npt.NDArray, method_name: str):
        if method_name not in self.method_names:
            self.file["attributions"].create_dataset(method_name, shape=(self.num_samples, *self.attributions_shape))
            self.method_names.append(method_name)
        self.file["attributions"][method_name][indices, ...] = attributions

    def __del__(self):
        self.file.close()
