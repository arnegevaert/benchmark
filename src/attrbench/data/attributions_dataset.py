from torch.utils.data import Dataset
import numpy as np
import h5py
from numpy import typing as npt
from typing import List


def _max_abs(arr: npt.NDArray, axis: int):
    return np.max(np.abs(arr), axis=axis, keepdims=True)

def _mean(arr: npt.NDArray, axis: int):
    return np.mean(arr, axis=axis, keepdims=True)


class AttributionsDataset(Dataset):
    """
    File
    - method_1: [num_samples, *sample_shape]
    - method_2: [num_samples, *sample_shape]
    - ...
    """

    def __init__(self, samples_dataset: Dataset, path: str, aggregate_axis: int = 0, aggregate_method: str = None,
                 methods: List[str] | None = None):
        self.path = path
        self.samples_dataset: Dataset = samples_dataset
        self.attributions_file: h5py.File | None = None
        self.aggregate_fn = None
        self.aggregate_axis = aggregate_axis
        if aggregate_method is not None:
            agg_fns = {
                "mean": _mean,
                "max_abs": _max_abs
            }
            self.aggregate_fn = agg_fns[aggregate_method]
        with h5py.File(path, "r") as fp:
            self.num_samples = fp.attrs["num_samples"]
            if methods is None:
                self.method_names = list(fp.keys())
            elif all(m in fp for m in methods):
                self.method_names = methods
            else:
                raise ValueError(f"Invalid methods: {methods}")

    def __getitem__(self, index):
        if self.attributions_file is None:
        #if not hasattr(self, "attributions_file"):
            self.attributions_file = h5py.File(self.path, "r")
        method_idx = index // self.num_samples
        method_name = self.method_names[method_idx]
        sample_idx = index % self.num_samples
        sample, label = self.samples_dataset[sample_idx]
        attrs = self.attributions_file[method_name][sample_idx]
        if self.aggregate_fn is not None:
            attrs = self.aggregate_fn(attrs, axis=self.aggregate_axis)
        return sample_idx, sample, label, attrs, method_name

    def __len__(self):
        return self.num_samples * len(self.method_names)
