import torch
from torch.utils.data import TensorDataset
from attribench.data._index_dataset import IndexDataset
from .._typing import _check_is_dataset
from torch.utils.data import Dataset
import numpy as np
import h5py
from typing import List, Dict, Tuple


def _max_abs(arr: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.max(torch.abs(arr), dim=dim, keepdim=True)


def _mean(arr: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.mean(arr, dim=dim, keepdim=True)


def _check_is_dataset_or_tensor(obj) -> h5py.Dataset | torch.Tensor:
    if isinstance(obj, h5py.Dataset) or isinstance(obj, torch.Tensor):
        return obj
    else:
        raise ValueError(
            f"Expected obj to be a Dataset or Tensor, but got {type(obj)}"
        )


def _get_attributions_shape(
    attributions: Dict[str, torch.Tensor] | h5py.File, method_names: List[str]
) -> Tuple[int, ...]:
    shape = None
    for m_name in method_names:
        if shape is None:
            # If shape is None, set it to the shape of the first method
            if isinstance(attributions, h5py.File):
                dataset = _check_is_dataset(attributions[m_name])
                shape = dataset.shape
            else:
                shape = attributions[m_name].shape
        else:
            # Otherwise, check if the shape for the current method
            # is the same as the first
            if isinstance(attributions, h5py.File):
                dataset = _check_is_dataset(attributions[m_name])
                cur_shape = dataset.shape
            else:
                cur_shape = attributions[m_name].shape
            if shape != cur_shape:
                raise ValueError(
                    "Attributions must have the same shape for each method"
                )
    if shape is None:
        raise ValueError("Attributions must not be empty")
    return shape


def _parse_attributions_dict(
    attributions: Dict[str, torch.Tensor],
    methods: List[str] | None,
) -> Tuple[List[str], int, Tuple[int, ...]]:
    if methods is None:
        method_names = list(attributions.keys())
    elif all(m in attributions.keys() for m in methods):
        method_names = methods
    else:
        raise ValueError(f"Invalid methods: {methods}")

    shape = _get_attributions_shape(attributions, method_names)

    num_samples = shape[0]
    attributions_shape = shape[1:]
    return method_names, num_samples, attributions_shape


def _parse_attributions_file(
    path: str, methods: List[str] | None
) -> Tuple[List[str], int, Tuple[int, ...]]:
    with h5py.File(path, "r") as fp:
        # Check if methods argument is valid
        if methods is None:
            method_names = list(fp.keys())
        elif all(m in fp for m in methods):
            method_names = methods
        else:
            raise ValueError(f"Invalid methods: {methods}")

        # Check if num_samples metadata is valid
        num_samples = fp.attrs["num_samples"]
        if isinstance(num_samples, np.integer):
            num_samples = int(num_samples)
        else:
            raise ValueError(
                f"Expected num_samples to be an integer,"
                f" but got {type(num_samples)}"
            )

        # Check if attributions have the same shape for each method
        shape = _get_attributions_shape(fp, method_names)
        attributions_shape = shape[1:]
    return method_names, num_samples, attributions_shape


class AttributionsDataset(IndexDataset):
    """
    (see functional interface for compute_attributions)
    File
    - method_1: ``[num_samples, *sample_shape]``
    - method_2: ``[num_samples, *sample_shape]``
    - ...
    """

    def __init__(
        self,
        samples: Dataset | torch.Tensor,
        labels: torch.Tensor | None = None,
        attributions: Dict[str, torch.Tensor] | None = None,
        methods: List[str] | None = None,
        path: str | None = None,
        aggregate_dim: int = 0,
        aggregate_method: str | None = None,
    ):
        self.path = path

        # If samples and labels are given as Tensors, wrap them in a Dataset
        self.samples_dataset: Dataset
        if isinstance(samples, torch.Tensor):
            if labels is None:
                raise ValueError(
                    "Labels must be given if samples are given as a Tensor"
                )
            self.samples_dataset = TensorDataset(samples, labels)
        else:
            self.samples_dataset = samples

        # Handle attributions dict or file
        self.attributions: Dict[str, torch.Tensor] | h5py.File | None = None
        self.method_names: List[str]
        orig_attributions_shape: Tuple[int, ...]
        if attributions is not None:
            # If attributions are given as a dict, parse the dict for metadata.
            (
                self.method_names,
                self.num_samples,
                orig_attributions_shape,
            ) = _parse_attributions_dict(attributions, methods)
            self.attributions = attributions
        else:
            # Otherwise, a path must be given. Load metadata from HDF5 file.
            if path is None:
                raise ValueError("Either attributions or path must be given")
            (
                self.method_names,
                self.num_samples,
                orig_attributions_shape,
            ) = _parse_attributions_file(path, methods)

        # Handle aggregation if necessary
        self.aggregate_fn = None
        self.aggregate_dim = aggregate_dim
        self.attributions_shape: Tuple[int, ...]
        if aggregate_method is not None:
            agg_fns = {"mean": _mean, "max_abs": _max_abs}
            self.aggregate_fn = agg_fns[aggregate_method]
        if self.aggregate_fn is not None:
            # If we aggregate over some axis, drop the corresponding axis
            self.attributions_shape = (
                orig_attributions_shape[: self.aggregate_dim]
                + orig_attributions_shape[self.aggregate_dim + 1 :]
            )
        else:
            self.attributions_shape = orig_attributions_shape

    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        if self.attributions is None:
            self.attributions = h5py.File(self.path, "r")
        method_idx = index // self.num_samples
        method_name = self.method_names[method_idx]
        sample_idx = index % self.num_samples
        sample, label = self.samples_dataset[sample_idx]
        dataset = _check_is_dataset_or_tensor(self.attributions[method_name])
        attrs = dataset[sample_idx]
        if not isinstance(attrs, torch.Tensor):
            attrs = torch.tensor(attrs)
        if self.aggregate_fn is not None:
            attrs = self.aggregate_fn(attrs, dim=self.aggregate_dim)
        return sample_idx, sample, label, attrs, method_name

    def __len__(self):
        return self.num_samples * len(self.method_names)


class _GroupedAttributionsDataset(Dataset):
    def __init__(self, dataset: AttributionsDataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        if self.dataset.attributions is None:
            raise ValueError("Attributions file not open")
        sample, label = self.dataset.samples_dataset[index]
        attrs: Dict[str, torch.Tensor] = {}
        for method_name in self.dataset.method_names:
            dataset = _check_is_dataset_or_tensor(
                self.dataset.attributions[method_name]
            )
            attrs[method_name] = dataset[index]
            if not isinstance(attrs[method_name], torch.Tensor):
                attrs[method_name] = torch.tensor(attrs[method_name])
        if self.dataset.aggregate_fn is not None:
            for method_name in self.dataset.method_names:
                attrs[method_name] = self.dataset.aggregate_fn(
                    attrs[method_name], dim=self.dataset.aggregate_dim
                )
        return index, sample, label, attrs

    def __len__(self):
        return self.dataset.num_samples
