from typing import Callable, List, Union, Tuple, Dict
from os import path
import os
import multiprocessing

import numpy as np
import torch

from attrbench.lib import AttributionWriter
from attrbench.metrics import Metric, MetricResult
from ._compute_perturbations import _compute_perturbations
from ._compute_result import _compute_result
from .result import InfidelityResult


def infidelity(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
               perturbation_mode: str, perturbation_size: float, num_perturbations: int,
               mode: Union[Tuple[str], str] = "mse", activation_fn: Union[Tuple[str], str] = "linear",
               writer: AttributionWriter = None) -> Dict:
    if type(activation_fn) == str:
        activation_fn = (activation_fn,)
    pert_vectors, pred_diffs = _compute_perturbations(samples, labels, model, perturbation_mode,
                                                      perturbation_size, num_perturbations, activation_fn, writer)
    if type(mode) == str:
        mode = (mode,)
    for m in mode:
        if m not in ("mse", "corr"):
            raise ValueError(f"Invalid mode: {m}")
    res = _compute_result(pert_vectors, pred_diffs, {"m": attrs}, mode)
    return res["m"]


class Infidelity(Metric):
    def __init__(self, model: Callable, method_names: List[str], perturbation_mode: str,
                 perturbation_size: float, num_perturbations: int, mode: Union[Tuple[str], str] = "mse",
                 activation_fn: Union[Tuple[str], str] = "linear", writer_dir: str = None):
        super().__init__(model, method_names)  # We don't pass writer_dir to super because we only use 1 general writer
        self.writers = {"general": AttributionWriter(path.join(writer_dir, "general"))} \
            if writer_dir is not None else None
        self.perturbation_mode = perturbation_mode
        self.perturbation_size = perturbation_size
        self.num_perturbations = num_perturbations
        self.mode = (mode,) if type(mode) == str else mode
        self.activation_fn = (activation_fn,) if type(activation_fn) == str else activation_fn
        self.result = InfidelityResult(method_names, perturbation_mode, perturbation_size,
                                       self.mode, self.activation_fn)
        self.pool = None

    def _append_cb(self, results):
        for method_name in results:
            self.result.append(method_name, results[method_name])

    def run_batch(self, samples, labels, attrs_dict: dict):
        if self.pool is not None:
            self.pool.join()
        # First calculate perturbation vectors and predictions differences, these can be re-used for all methods
        writer = self.writers["general"] if self.writers is not None else None
        pert_vectors, pred_diffs = _compute_perturbations(samples, labels, self.model,
                                                          self.perturbation_mode, self.perturbation_size,
                                                          self.num_perturbations, self.activation_fn, writer)
        if os.getenv("NO_MULTIPROC"):
            results = _compute_result(pert_vectors, pred_diffs, attrs_dict, self.mode)
            for method_name in results:
                self.result.append(method_name, results[method_name])
        else:
            self.pool = multiprocessing.Pool(processes=1)
            self.pool.apply_async(_compute_result, args=(pert_vectors, pred_diffs, attrs_dict, self.mode),
                                  callback=self._append_cb)
            self.pool.close()

    def get_result(self) -> MetricResult:
        if self.pool is not None:
            self.pool.join()
        return self.result
