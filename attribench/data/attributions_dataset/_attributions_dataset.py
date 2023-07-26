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
    Represents a dataset containing attributions for a set of samples and
    attribution methods.

    The samples and labels can be given in two ways. Either a PyTorch
    ``Dataset`` is passed to the ``samples`` argument
    containing both the samples and the labels, or a Tensor is passed to the
    ``samples`` argument and a Tensor is passed to the ``labels`` argument.

    An AttributionsDataset can be constructed from a dictionary of attributions
    or from an HDF5 file containing the attributions. 
    If attributions are given using a dictionary, the dictionary must map
    method names to Tensors containing the attributions for each sample. The
    attributions must have the same shape for each method. The shape of the
    attributions must be ``[num_samples, *sample_shape]``.

    If attributions are given using an HDF5 file, the file must contain a
    dataset for each attribution method. The dataset must have the same shape
    for each method. The shape of the dataset must be
    ``[num_samples, *sample_shape]``. The file must also contain an attribute
    ``num_samples`` specifying the number of samples in the dataset.

    A list of method names can be given using the ``methods`` argument. If
    ``methods`` is None, all methods in the attributions dictionary or file
    are used. Otherwise, only the methods in the ``methods`` list are used.

    Attributions can be aggregated over some dimension by specifying the
    aggregate_dim and aggregate_method arguments. The aggregate_dim argument
    specifies the dimension over which to aggregate. The aggregate_method
    argument specifies the method to use for aggregation. The aggregate_method
    argument must be one of ``"mean"`` or ``"max_abs"``. Note that the
    aggregate_dim argument is specified in terms of the shape of the
    attributions, i.e. excluding the ``num_samples`` dimension.
    
    For example, if the
    attributions have shape ``[num_samples, 3, 32, 32]``, then the
    attributions can be aggregated over the channel dimension by setting
    ``aggregate_dim=0``. The resulting attributions will have shape
    ``[num_samples, 32, 32]``.
    """

    def __init__(
        self,
        samples: Dataset | torch.Tensor,
        labels: torch.Tensor | None = None,
        path: str | None = None,
        attributions: Dict[str, torch.Tensor] | None = None,
        methods: List[str] | None = None,
        aggregate_dim: int = 0,
        aggregate_method: str | None = None,
    ):
        """
        Parameters
        ----------
        samples: Dataset | torch.Tensor
            A Dataset containing samples and labels, or a Tensor containing the
            samples for which attributions are given.
        labels: torch.Tensor | None
            A Tensor containing the labels for the samples.
            Only used if samples is a Tensor.
        path: str | None
            Path to an HDF5 file containing the attributions.
            If None, attributions must be given as a dictionary.
        attributions: Dict[str, torch.Tensor] | None
            A dictionary mapping attribution method names to Tensors containing
            the attributions for each sample. If None, a path to an HDF5 file
            must be given.
        methods: List[str] | None
            A list of method names to use. If None, all methods in the
            attributions dictionary are used.
        aggregate_dim: int
            If not None, aggregate the attributions over the given dimension.
        aggregate_method: str | None
            If not None, aggregate the attributions using the given method.
            Must be one of "mean" or "max_abs" or None.
        
        Raises
        ------
        ValueError
            If attributions is None and path is None, or if labels is None and
            samples is a Tensor.
        """
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
        
    def _open_attributions_file(self):
        self.attributions = h5py.File(self.path, "r")

    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        if self.attributions is None:
            self._open_attributions_file()
        assert self.attributions is not None
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


class GroupedAttributionsDataset(IndexDataset):
    def __init__(self, dataset: AttributionsDataset):
        super().__init__(dataset)
        self.dataset: AttributionsDataset = dataset
        self.method_names = dataset.method_names
        self.num_samples = dataset.num_samples
        self.attributions_shape = dataset.attributions_shape

    def __getitem__(self, index):
        if self.dataset.attributions is None:
            self.dataset._open_attributions_file()
            assert self.dataset.attributions is not None
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
