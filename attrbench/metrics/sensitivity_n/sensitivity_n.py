from typing import Callable, List, Dict, Union, Tuple
from os import path

import numpy as np
import torch

from attrbench.lib import segment_attributions, AttributionWriter
from attrbench.lib.masking import Masker
from attrbench.metrics import Metric
from ._compute_correlations import _compute_correlations
from ._compute_perturbations import _compute_perturbations
from ._dataset import _SensitivityNDataset, _SegSensNDataset
from .result import SegSensitivityNResult, SensitivityNResult


def sensitivity_n(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
                  min_subset_size: float, max_subset_size: float, num_steps: int, num_subsets: int,
                  masker: Masker, activation_fn: Union[Tuple[str], str] = "linear",
                  writer: AttributionWriter = None):
    if type(activation_fn) == str:
        activation_fn = (activation_fn,)
    num_features = attrs.reshape(attrs.shape[0], -1).shape[1]
    n_range = (np.linspace(min_subset_size, max_subset_size, num_steps) * num_features).astype(np.int)
    ds = _SensitivityNDataset(n_range, num_subsets, samples.cpu().numpy(), num_features, masker)
    output_diffs, indices = _compute_perturbations(samples, labels, ds, model, n_range, activation_fn, writer)
    return _compute_correlations(attrs, n_range, output_diffs, indices)


def seg_sensitivity_n(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
                      min_subset_size: float, max_subset_size: float, num_steps: int, num_subsets: int,
                      masker: Masker, activation_fn: Union[Tuple[str], str] = "linear",
                      writer: AttributionWriter = None):
    # Total number of segments is fixed 100
    if type(activation_fn) == str:
        activation_fn = (activation_fn,)
    n_range = (np.linspace(min_subset_size, max_subset_size, num_steps) * 100).astype(np.int)
    ds = _SegSensNDataset(n_range, num_subsets, samples.cpu().numpy(), masker, writer)
    attrs = segment_attributions(ds.segmented_images, attrs)
    output_diffs, indices = _compute_perturbations(samples, labels, ds, model, n_range, activation_fn, writer)
    return _compute_correlations(attrs, n_range, output_diffs, indices)


class SensitivityN(Metric):
    def __init__(self, model: Callable, method_names: List[str], min_subset_size: float, max_subset_size: float,
                 num_steps: int, num_subsets: int, masker: Masker, activation_fn: Union[Tuple[str], str],
                 writer_dir: str = None):
        super().__init__(model, method_names, writer_dir)
        self.min_subset_size = min_subset_size
        self.max_subset_size = max_subset_size
        self.num_steps = num_steps
        self.num_subsets = num_subsets
        self.masker = masker
        self.activation_fn = (activation_fn,) if type(activation_fn) == str else activation_fn
        self.result = SensitivityNResult(method_names, self.activation_fn,
                                         index=np.linspace(min_subset_size, max_subset_size, num_steps))
        if self.writer_dir is not None:
            self.writers["general"] = AttributionWriter(self.writer_dir)

    def run_batch(self, samples, labels, attrs_dict: Dict[str, np.ndarray]):
        # Get total number of features from attributions dict
        attrs = attrs_dict[next(iter(attrs_dict))]
        num_features = attrs.reshape(attrs.shape[0], -1).shape[1]
        # Calculate n_range
        n_range = (np.linspace(self.min_subset_size, self.max_subset_size, self.num_steps) * num_features).astype(
            np.int)
        # Create pseudo-dataset
        ds = _SensitivityNDataset(n_range, self.num_subsets, samples.cpu().numpy(), num_features, self.masker)
        # Calculate output diffs and removed indices (we will re-use this for each method)
        writer = self.writers["general"] if self.writers is not None else None
        output_diffs, indices = _compute_perturbations(samples, labels, ds, self.model, n_range, self.activation_fn,
                                                       writer)

        for method_name in attrs_dict:
            attrs = attrs_dict[method_name]
            attrs = attrs.reshape((attrs.shape[0], 1, -1))  # [batch_size, 1, -1]
            self.result.append(method_name, _compute_correlations(attrs, n_range, output_diffs, indices))


class SegSensitivityN(Metric):
    def __init__(self, model: Callable, method_names: List[str], min_subset_size: float, max_subset_size: float,
                 num_steps: int, num_subsets: int, masker: Masker, activation_fn: Union[Tuple[str], str],
                 writer_dir: str = None):
        super().__init__(model, method_names, writer_dir)
        self.min_subset_size = min_subset_size
        self.max_subset_size = max_subset_size
        self.num_steps = num_steps
        self.num_subsets = num_subsets
        # Total number of segments is fixed 100
        self.n_range = (np.linspace(self.min_subset_size, self.max_subset_size, self.num_steps) * 100).astype(np.int)
        self.masker = masker
        self.activation_fn = (activation_fn,) if type(activation_fn) == str else activation_fn
        self.result = SegSensitivityNResult(method_names, self.activation_fn,
                                            index=np.linspace(min_subset_size, max_subset_size, num_steps))
        if self.writer_dir is not None:
            self.writers["general"] = AttributionWriter(path.join(self.writer_dir, "general"))

    def run_batch(self, samples, labels, attrs_dict: dict):
        # Create pseudo-dataset
        ds = _SegSensNDataset(self.n_range, self.num_subsets, samples.cpu().numpy(), self.masker)
        # Calculate output diffs and removed indices (we will re-use this for each method)
        writer = self.writers["general"] if self.writers is not None else None
        output_diffs, indices = _compute_perturbations(samples, labels, ds, self.model, self.n_range,
                                                       self.activation_fn, writer)

        for method_name in attrs_dict:
            attrs = segment_attributions(ds.segmented_images, attrs_dict[method_name])
            self.result.append(method_name, _compute_correlations(attrs, self.n_range, output_diffs, indices))
