from attribench.data._index_dataset import IndexDataset
from torch.utils.data import Dataset
import numpy as np
import h5py
from numpy import typing as npt
from typing import Tuple, Optional


def _max_abs(arr: npt.NDArray, axis: int):
    return np.max(np.abs(arr), axis=axis, keepdims=True)


def _mean(arr: npt.NDArray, axis: int):
    return np.mean(arr, axis=axis, keepdims=True)


class AttributionsDataset(IndexDataset):
    """
    TODO create AttributionsDataset from dict of attributions
    (see functional interface for compute_attributions)
    File
    ```
    - method_1: [num_samples, *sample_shape]
    - method_2: [num_samples, *sample_shape]
    - ...
    ```
    """

    def __init__(
        self,
        samples_dataset: Dataset,
        path: str,
        aggregate_axis: int = 0,
        aggregate_method: Optional[str] = None,
        methods: Optional[Tuple[str]] = None,
        group_attributions=False,
    ):
        self.path = path
        # If True, all attributions for a given sample are returned for a given sample.
        # This can be useful if some intermediate results using the sample can be re-used (e.g. in Infidelity)
        self.group_attributions = group_attributions
        self.samples_dataset: Dataset = samples_dataset
        self.attributions_file: Optional[h5py.File] = None
        self.aggregate_fn = None
        self.aggregate_axis = aggregate_axis
        if aggregate_method is not None:
            agg_fns = {"mean": _mean, "max_abs": _max_abs}
            self.aggregate_fn = agg_fns[aggregate_method]
        self.method_names: Tuple[str]
        with h5py.File(path, "r") as fp:
            num_samples = fp.attrs["num_samples"]
            if isinstance(num_samples, np.int64):
                self.num_samples: int = int(num_samples)
            if methods is None:
                self.method_names = tuple(fp.keys())
            elif all(m in fp for m in methods):
                self.method_names = methods
            else:
                raise ValueError(f"Invalid methods: {methods}")
            orig_attributions_shape = fp[self.method_names[0]].shape[1:]
            if self.aggregate_fn is not None:
                self.attributions_shape = (
                    orig_attributions_shape[: self.aggregate_axis]
                    + orig_attributions_shape[self.aggregate_axis + 1 :]
                )
            else:
                self.attributions_shape = orig_attributions_shape

    def get_item_nongrouped(self, index):
        method_idx = index // self.num_samples
        method_name = self.method_names[method_idx]
        sample_idx = index % self.num_samples
        sample, label = self.samples_dataset[sample_idx]
        attrs = self.attributions_file[method_name][sample_idx]
        if self.aggregate_fn is not None:
            attrs = self.aggregate_fn(attrs, axis=self.aggregate_axis)
        return sample_idx, sample, label, attrs, method_name

    def get_item_grouped(self, index):
        sample, label = self.samples_dataset[index]
        attrs = {
            method_name: self.attributions_file[method_name][index]
            for method_name in self.method_names
        }
        if self.aggregate_fn is not None:
            for method_name in self.method_names:
                attrs[method_name] = self.aggregate_fn(
                    attrs[method_name], axis=self.aggregate_axis
                )
        return index, sample, label, attrs

    def __getitem__(self, index):
        if self.attributions_file is None:
            self.attributions_file = h5py.File(self.path, "r")
        if self.group_attributions:
            return self.get_item_grouped(index)
        return self.get_item_nongrouped(index)

    def __len__(self):
        if self.group_attributions:
            return self.num_samples
        return self.num_samples * len(self.method_names)
