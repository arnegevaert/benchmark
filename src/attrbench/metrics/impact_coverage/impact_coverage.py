from os import path
from typing import Callable, Dict

import torch

from attrbench.metrics import Metric
from ._compute_coverage import _compute_coverage
from ._apply_patches import _apply_patches
from .result import ImpactCoverageResult
import numpy as np


def impact_coverage(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
                    patch_folder: str, writer=None):
    if len(samples.shape) != 4:
        raise ValueError("Impact Coverage can only be computed for image data and expects 4 input dimensions")
    attacked_samples, patch_mask, targets = _apply_patches(samples, labels, model, patch_folder)
    return _compute_coverage(attacked_samples, patch_mask, targets, method, writer)


class ImpactCoverage(Metric):
    def __init__(self, model, methods: Dict[str, Callable], patch_folder: str, writer_dir: str = None):
        self.methods = methods
        super().__init__(model, list(methods.keys()), writer_dir)
        self.patch_folder = patch_folder
        self._result: ImpactCoverageResult = ImpactCoverageResult(list(methods.keys()) + ["_BASELINE"])

    def run_batch(self, samples, labels, attrs_dict: Dict = None, baseline_attrs: np.ndarray = None):
        attacked_samples, patch_mask, targets = _apply_patches(samples, labels,
                                                               self.model, self.patch_folder)
        batch_result = {}
        # Compute results on baseline attributions
        baseline_result = []
        for i in range(baseline_attrs.shape[0]):
            baseline_result.append(
                _compute_coverage(attacked_samples, patch_mask, targets, attrs=baseline_attrs[i, ...]).reshape(-1, 1))
        batch_result["_BASELINE"] = np.stack(baseline_result, axis=1)

        # Compute results on actual attributions
        for method_name in self.methods:
            batch_result[method_name] = _compute_coverage(attacked_samples, patch_mask, targets,
                                                          self.methods[method_name],
                                                          writer=self._get_writer(method_name)).reshape(-1, 1).cpu().detach().numpy()
        self.result.append(batch_result)
