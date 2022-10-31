from torch.utils.data import Dataset
import numpy as np
import h5py
from numpy import typing as npt
from typing import List


def _max_abs(arr: npt.NDArray, axis: int):
    return np.max(np.abs(arr), axis=axis)


class AttributionsDataset(Dataset):
    """
    File
    - methods
        - method_1: [num_samples, *sample_shape]
        - method_2: [num_samples, *sample_shape]
        - ...
    - baselines
        - baseline_1: [num_samples, *sample_shape]
        - baseline_2: [num_samples, *sample_shape]
        - ...
    """

    def __init__(self, samples_dataset: Dataset, path: str, aggregate_axis: int = 0, aggregate_method: str = None,
                 baselines: List[str] | None = None, methods: List[str] | None = None):
        self.path = path
        self.samples_dataset: Dataset = samples_dataset
        self.attributions_file: h5py.File | None = None
        self.aggregate_fn = None
        self.aggregate_axis = aggregate_axis
        if aggregate_method is not None:
            agg_fns = {
                "mean": np.mean,
                "max_abs": _max_abs
            }
            self.aggregate_fn = agg_fns[aggregate_method]
        with h5py.File(path, "r") as fp:
            self.num_samples = fp.attrs["num_samples"]

            if methods is None:
                self.attribution_methods = list(fp["methods"].keys())
            elif all(m in fp["methods"] for m in methods):
                self.attribution_methods = methods
            else:
                raise ValueError(f"Invalid methods: {methods}")

            if baselines is None:
                self.baselines = list(fp["baselines"].keys())
            elif all(b in fp["baselines"] for b in baselines):
                self.baselines = baselines
            else:
                raise ValueError(f"Invalid baselines: {baselines}")

    def __getitem__(self, index):
        if self.attributions_file is None:
            self.attributions_file = h5py.File(self.path, "r")
        methods_baselines = self.attribution_methods + self.baselines
        method_baseline_idx = index // self.num_samples
        is_baseline = method_baseline_idx >= len(self.attribution_methods)
        method_name = methods_baselines[method_baseline_idx]
        sample_idx = index % self.num_samples
        sample, label = self.samples_dataset[sample_idx]
        upper_key = "baselines" if is_baseline else "methods"
        attrs = self.attributions_file[upper_key][method_name][sample_idx]
        if self.aggregate_fn is not None:
            attrs = self.aggregate_fn(attrs, axis=self.aggregate_axis)
        return sample_idx, sample, label, attrs, method_name, is_baseline

    def __len__(self):
        if self.attributions_file is None:
            self.attributions_file = h5py.File(self.path, "r")
        return self.num_samples * (len(self.attribution_methods) + len(self.baselines))
