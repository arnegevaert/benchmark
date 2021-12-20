from typing import Callable, List, Dict, Union, Tuple
import multiprocessing
from os import path
import os

import numpy as np
import torch

from attrbench.lib import AttributionWriter, segment_attributions
from attrbench.lib.masking import Masker, ImageMasker
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

    ds = _SensitivityNDataset(n_range, num_subsets, samples, masker)

    output_diffs, indices = _compute_perturbations(samples, labels, ds, model, n_range, activation_fn, writer)
    return _compute_correlations(attrs, n_range, output_diffs, indices)


def seg_sensitivity_n(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
                      min_subset_size: float, max_subset_size: float, num_steps: int, num_subsets: int,
                      masker: ImageMasker, activation_fn: Union[Tuple[str], str] = "linear",
                      writer: AttributionWriter = None):
    # Total number of segments is fixed 100
    if type(activation_fn) == str:
        activation_fn = (activation_fn,)
    n_range = (np.linspace(min_subset_size, max_subset_size, num_steps) * 100).astype(np.int)
    ds = _SegSensNDataset(n_range, num_subsets, samples, writer)
    ds.set_masker(masker)

    # TODO using tensors here could improve GPU usage
    attrs = segment_attributions(ds.segmented_images.cpu().numpy(), attrs)

    output_diffs, indices = _compute_perturbations(samples, labels, ds, model, n_range, activation_fn, writer)
    return _compute_correlations(attrs, n_range, output_diffs, indices)


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
        self._result: SensitivityNResult = SensitivityNResult(method_names + ["_BASELINE"], list(self.maskers.keys()),
                                                              list(self.activation_fns),
                                                              index=np.linspace(min_subset_size, max_subset_size,
                                                                                num_steps))
        self.pool = None

    def run_batch(self, samples, labels, attrs_dict: Dict[str, np.ndarray], baseline_attrs: np.ndarray):
        # Get total number of features from attributions dict
        attrs = attrs_dict[next(iter(attrs_dict))]
        num_features = attrs.reshape(attrs.shape[0], -1).shape[1]
        # Calculate n_range
        n_range = (np.linspace(self.min_subset_size, self.max_subset_size, self.num_steps) * num_features).astype(
            np.int)
        writer = self.writers["general"] if self.writers is not None else None

        output_diffs_dict, indices_dict = {}, {}
        for masker_name, masker in self.maskers.items():
            # Create pseudo-dataset
            ds = _SensitivityNDataset(n_range, self.num_subsets, samples, masker)
            # Calculate output diffs and removed indices (we will re-use this for each method)
            output_diffs, indices = _compute_perturbations(samples, labels, ds, self.model, n_range,
                                                           self.activation_fns,
                                                           writer)
            output_diffs_dict[masker_name] = output_diffs
            indices_dict[masker_name] = indices

        if os.getenv("NO_MULTIPROC"):
            self.compute_and_append_results(n_range, output_diffs_dict, indices_dict, attrs_dict, baseline_attrs, self.result)
        else:
            result = self.result
            self.pool = multiprocessing.pool.ThreadPool(processes=1)
            self.pool.apply_async(
                self.compute_and_append_results,
                args=(n_range, output_diffs_dict, indices_dict, attrs_dict, baseline_attrs, result)
            )
            self.pool.close()

    def compute_and_append_results(self, n_range: List[int], output_diffs_dict: Dict, indices_dict: Dict,
                                   attrs_dict: Dict[str, np.ndarray], baseline_attrs: np.ndarray,
                                   result: SensitivityNResult):
        method_results = {masker: {afn: {method_name: None for method_name in self.method_names}
                                   for afn in self.activation_fns}
                          for masker in self.maskers}
        baseline_results = {masker: {afn: [] for afn in self.activation_fns} for masker in self.maskers}
        for masker_name in self.maskers:
            for method_name in self.method_names:
                res = _compute_correlations(attrs_dict[method_name], n_range,
                                            output_diffs_dict[masker_name],
                                            indices_dict[masker_name])
                for afn in self.activation_fns:
                    method_results[masker_name][afn][method_name] = res[afn].cpu().detach().numpy()

            for i in range(baseline_attrs.shape[0]):
                res = _compute_correlations(baseline_attrs[i, ...], n_range, output_diffs_dict[masker_name],
                                            indices_dict[masker_name])
                for afn in self.activation_fns:
                    baseline_results[masker_name][afn].append(res[afn])
            for afn in self.activation_fns:
                baseline_results[masker_name][afn] = np.stack(baseline_results[masker_name][afn], axis=1)
        result.append(method_results)
        result.append(baseline_results, method="_BASELINE")
        logging.info("Appended Sensitivity-n")

    @property
    def result(self) -> SensitivityNResult:
        if self.pool is not None:
            start_t = time.time()
            logging.info("Joining Sensitivity-N...")
            self.pool.join()
            end_t = time.time()
            logging.info(f"Join done in {end_t - start_t:.2f}s")
        return self._result


class SegSensitivityN(SensitivityN):
    def __init__(self, model: Callable, method_names: List[str], min_subset_size: float, max_subset_size: float,
                 num_steps: int, num_subsets: int, maskers: Dict, activation_fns: Union[Tuple[str], str],
                 writer_dir: str = None):
        super().__init__(model, method_names, min_subset_size, max_subset_size, num_steps, num_subsets, maskers,
                         activation_fns, writer_dir)
        # Total number of segments is fixed 100
        self.n_range = (np.linspace(self.min_subset_size, self.max_subset_size, self.num_steps) * 100).astype(np.int)
        self._result: SegSensitivityNResult = SegSensitivityNResult(method_names + ["_BASELINE"],
                                                                    list(self.maskers.keys()),
                                                                    list(self.activation_fns),
                                                                    index=np.linspace(min_subset_size, max_subset_size,
                                                                                      num_steps))
        self.pool = None

    def run_batch(self, samples, labels, attrs_dict: dict, baseline_attrs: np.ndarray):
        writer = self.writers["general"] if self.writers is not None else None
        ds = _SegSensNDataset(self.n_range, self.num_steps, samples, writer)
        # TODO using tensors here could improve GPU usage
        segmented_attrs_dict = {method_name: segment_attributions(ds.segmented_images.cpu().numpy(),
                                                                  attrs_dict[method_name])
                                for method_name in attrs_dict}
        segmented_baseline_attrs = np.stack(
            [segment_attributions(ds.segmented_images.cpu().numpy(), baseline_attrs[i, ...])
             for i in range(baseline_attrs.shape[0])], axis=0)

        output_diffs_dict, indices_dict = {}, {}
        for masker_name, masker in self.maskers.items():
            ds.set_masker(masker)
            # Calculate output diffs and removed indices (we will re-use this for each method)
            output_diffs, indices = _compute_perturbations(samples, labels, ds, self.model, self.n_range,
                                                           self.activation_fns, writer)
            output_diffs_dict[masker_name] = output_diffs
            indices_dict[masker_name] = indices
        if os.getenv("NO_MULTIPROC"):
            self.compute_and_append_results(self.n_range, output_diffs_dict, indices_dict, segmented_attrs_dict,
                                            segmented_baseline_attrs, self.result)
        else:
            result = self.result
            self.pool = multiprocessing.pool.ThreadPool(processes=1)
            self.pool.apply_async(self.compute_and_append_results,
                                  args=(
                                      self.n_range, output_diffs_dict, indices_dict, segmented_attrs_dict,
                                      segmented_baseline_attrs, result))
            self.pool.close()

    @property
    def result(self) -> SegSensitivityNResult:
        if self.pool is not None:
            start_t = time.time()
            logging.info("Joining Seg-Sensitivity-N...")
            self.pool.join()
            end_t = time.time()
            logging.info(f"Join done in {end_t - start_t:.2f}s")
        return self._result
