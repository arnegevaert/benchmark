from typing import Callable, List, Dict, Union, Tuple
import multiprocessing
from os import path
import os

import numpy as np
import torch

from attrbench.lib import segment_attributions, AttributionWriter
from attrbench.lib.masking import Masker
from attrbench.metrics import MaskerMetric
from ._compute_correlations import _compute_correlations
from ._compute_perturbations import _compute_perturbations
from ._dataset import _SensitivityNDataset, _SegSensNDataset
from .result import SegSensitivityNResult, SensitivityNResult
import logging
import time


def sensitivity_n(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
                  min_subset_size: float, max_subset_size: float, num_steps: int, num_subsets: int,
                  masker: Masker, activation_fn: Union[Tuple[str], str] = "linear",
                  writer: AttributionWriter = None):
    if type(activation_fn) == str:
        activation_fn = (activation_fn,)
    num_features = attrs.reshape(attrs.shape[0], -1).shape[1]
    n_range = (np.linspace(min_subset_size, max_subset_size, num_steps) * num_features).astype(np.int)
    ds = _SensitivityNDataset(n_range, num_subsets, samples, num_features, masker)
    output_diffs, indices = _compute_perturbations(samples, labels, ds, model, n_range, activation_fn, writer)
    res = _compute_correlations({"m": attrs}, n_range, output_diffs, indices)
    return res["m"]


def seg_sensitivity_n(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
                      min_subset_size: float, max_subset_size: float, num_steps: int, num_subsets: int,
                      masker: Masker, activation_fn: Union[Tuple[str], str] = "linear",
                      writer: AttributionWriter = None):
    # Total number of segments is fixed 100
    if type(activation_fn) == str:
        activation_fn = (activation_fn,)
    n_range = (np.linspace(min_subset_size, max_subset_size, num_steps) * 100).astype(np.int)
    ds = _SegSensNDataset(n_range, num_subsets, samples, masker, writer)
    attrs = segment_attributions(ds.segmented_images, attrs)
    output_diffs, indices = _compute_perturbations(samples, labels, ds, model, n_range, activation_fn, writer)
    res = _compute_correlations({"m": attrs}, n_range, output_diffs, indices)
    return res["m"]


class SensitivityN(MaskerMetric):
    def __init__(self, model: Callable, method_names: List[str], min_subset_size: float, max_subset_size: float,
                 num_steps: int, num_subsets: int, maskers: Dict, activation_fns: Union[Tuple[str], str],
                 writer_dir: str = None):
        super().__init__(model, method_names,
                         maskers)  # We don't pass writer_dir to super because we only use 1 general writer
        self.writers = {"general": AttributionWriter(path.join(writer_dir, "general"))} \
            if writer_dir is not None else None
        self.min_subset_size = min_subset_size
        self.max_subset_size = max_subset_size
        self.num_steps = num_steps
        self.num_subsets = num_subsets
        self.activation_fns = (activation_fns,) if type(activation_fns) == str else activation_fns
        self.result: SensitivityNResult = SensitivityNResult(method_names, list(self.maskers.keys()),
                                                             list(self.activation_fns),
                                                             index=np.linspace(min_subset_size, max_subset_size,
                                                                               num_steps))
        self.pool = None

    def append_result(self, masker_name, results):
        logging.info("Appending Sensitivity-N")
        for method_name in results:
            for afn in results[method_name]:
                self.result.append(method_name, masker_name, afn,
                                   results[method_name][masker_name].detach().cpu().numpy())

    def run_batch(self, samples, labels, attrs_dict: Dict[str, np.ndarray]):
        if self.pool is not None:
            logging.info("Joining Sensitivity-N...")
            self.pool.join()
            logging.info("Join done in {end_t - start_t:.2f}s")
        # Get total number of features from attributions dict
        attrs = attrs_dict[next(iter(attrs_dict))]
        num_features = attrs.reshape(attrs.shape[0], -1).shape[1]
        # Calculate n_range
        n_range = (np.linspace(self.min_subset_size, self.max_subset_size, self.num_steps) * num_features).astype(
            np.int)
        writer = self.writers["general"] if self.writers is not None else None
        for masker_name, masker in self.maskers.items():
            # Create pseudo-dataset
            ds = _SensitivityNDataset(n_range, self.num_subsets, samples, num_features, masker)
            # Calculate output diffs and removed indices (we will re-use this for each method)
            output_diffs, indices = _compute_perturbations(samples, labels, ds, self.model, n_range,
                                                           self.activation_fns,
                                                           writer)

            if os.getenv("NO_MULTIPROC"):
                results = _compute_correlations(attrs_dict, n_range, output_diffs, indices)
                self.append_result(masker_name, results)
            else:
                self.pool = multiprocessing.pool.ThreadPool(processes=1)
                self.pool.apply_async(_compute_correlations, args=(attrs_dict, n_range, output_diffs, indices),
                                      callback=lambda res: self.append_result(masker_name, res))
                self.pool.close()

    def get_result(self) -> SensitivityNResult:
        if self.pool is not None:
            start_t = time.time()
            logging.info("Joining Sensitivity-N...")
            self.pool.join()
            end_t = time.time()
            logging.info(f"Join done in {end_t - start_t:.2f}s")
        return self.result


class SegSensitivityN(MaskerMetric):
    def __init__(self, model: Callable, method_names: List[str], min_subset_size: float, max_subset_size: float,
                 num_steps: int, num_subsets: int, maskers: Dict, activation_fns: Union[Tuple[str], str],
                 writer_dir: str = None):
        super().__init__(model, method_names,
                         maskers)  # We don't pass writer_dir to super because we only use 1 general writer
        self.writers = {"general": AttributionWriter(path.join(writer_dir, "general"))} \
            if writer_dir is not None else None
        self.min_subset_size = min_subset_size
        self.max_subset_size = max_subset_size
        self.num_steps = num_steps
        self.num_subsets = num_subsets
        # Total number of segments is fixed 100
        self.n_range = (np.linspace(self.min_subset_size, self.max_subset_size, self.num_steps) * 100).astype(np.int)
        self.masker = maskers
        self.activation_fns = (activation_fns,) if type(activation_fns) == str else activation_fns
        self.result: SegSensitivityNResult = SegSensitivityNResult(method_names, list(self.maskers.keys()),
                                                                   list(self.activation_fns),
                                                                   index=np.linspace(min_subset_size, max_subset_size,
                                                                                     num_steps))
        self.pool = None

    def append_results(self, masker_name, results):
        logging.info("Appending Seg-Sensitivity-N")
        for method_name in results:
            for afn in results[method_name]:
                self.result.append(method_name, masker_name, afn,
                                   results[method_name][masker_name].detach().cpu().numpy())

    def run_batch(self, samples, labels, attrs_dict: dict):
        if self.pool is not None:
            start_t = time.time()
            logging.info("Joining Seg-Sensitivity-N...")
            self.pool.join()
            end_t = time.time()
            logging.info(f"Join done in {end_t - start_t:.2f}s")
        writer = self.writers["general"] if self.writers is not None else None
        for masker_name, masker in self.maskers.items():
            # Create pseudo-dataset
            ds = _SegSensNDataset(self.n_range, self.num_subsets, samples, masker)
            # Calculate output diffs and removed indices (we will re-use this for each method)
            output_diffs, indices = _compute_perturbations(samples, labels, ds, self.model, self.n_range,
                                                           self.activation_fns, writer)
            segmented_attrs_dict = {key: segment_attributions(ds.segmented_images,
                                                              torch.tensor(attrs_dict[key],
                                                                           device=samples.device)).cpu().numpy() for key
                                    in attrs_dict}

            if os.getenv("NO_MULTIPROC"):
                results = _compute_correlations(segmented_attrs_dict, self.n_range, output_diffs, indices)
                self.append_results(masker_name, results)
            else:
                self.pool = multiprocessing.pool.ThreadPool(processes=1)
                self.pool.apply_async(_compute_correlations,
                                      args=(segmented_attrs_dict, self.n_range, output_diffs, indices),
                                      callback=lambda res: self.append_results(masker_name, res))
                self.pool.close()

    def get_result(self) -> SegSensitivityNResult:
        if self.pool is not None:
            start_t = time.time()
            logging.info("Joining Seg-Sensitivity-N...")
            self.pool.join()
            end_t = time.time()
            logging.info(f"Join done in {end_t - start_t:.2f}s")
        return self.result
