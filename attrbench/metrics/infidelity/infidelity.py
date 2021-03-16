from typing import Callable, List, Union, Tuple, Dict
from os import path

import numpy as np
import torch

from attrbench.lib import AttributionWriter
from attrbench.metrics import Metric
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
    return _compute_result(pert_vectors, pred_diffs, attrs, mode)


class Infidelity(Metric):
    def __init__(self, model: Callable, method_names: List[str], perturbation_mode: str,
                 perturbation_size: float, num_perturbations: int, mode: Union[Tuple[str], str] = "mse",
                 activation_fn: Union[Tuple[str], str] = "linear", writer_dir: str = None, num_workers=0):
        super().__init__(model, method_names, writer_dir, num_workers)
        self.perturbation_mode = perturbation_mode
        self.perturbation_size = perturbation_size
        self.num_perturbations = num_perturbations
        self.mode = (mode,) if type(mode) == str else mode
        self.activation_fn = (activation_fn,) if type(activation_fn) == str else activation_fn
        if self.writer_dir is not None:
            self.writers["general"] = AttributionWriter(path.join(self.writer_dir, "general"))
        self.result = InfidelityResult(method_names, perturbation_mode, perturbation_size,
                                       self.mode, self.activation_fn)

    def run_batch(self, samples, labels, attrs_dict: dict):
        # First calculate perturbation vectors and predictions differences, these can be re-used for all methods
        writer = self.writers["general"] if self.writers is not None else None
        pert_vectors, pred_diffs = _compute_perturbations(samples, labels, self.model,
                                                          self.perturbation_mode, self.perturbation_size,
                                                          self.num_perturbations, self.activation_fn, writer,
                                                          self.num_workers)
        for method_name in attrs_dict:
            self.result.append(method_name, _compute_result(pert_vectors, pred_diffs, attrs_dict[method_name],
                                                            self.mode))


