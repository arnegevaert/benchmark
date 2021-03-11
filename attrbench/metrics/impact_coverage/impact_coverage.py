from os import path
from typing import Callable, Dict

import torch

from attrbench.metrics import Metric
from ._compute_coverage import _compute_coverage
from ._apply_patches import _apply_patches
from .result import ImpactCoverageResult


def impact_coverage(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
                    patch_folder: str, writer=None):
    if len(samples.shape) != 4:
        raise ValueError("Impact Coverage can only be computed for image data and expects 4 input dimensions")
    attacked_samples, patch_mask, targets = _apply_patches(samples, labels, model, patch_folder)
    return _compute_coverage(attacked_samples, method, patch_mask, targets, writer)


class ImpactCoverage(Metric):
    def __init__(self, model, methods: Dict[str, Callable], patch_folder: str, writer_dir: str = None):
        self.methods = methods
        super().__init__(model, list(methods.keys()), writer_dir)
        self.patch_folder = patch_folder
        self.writers = {method_name: path.join(writer_dir, method_name) if writer_dir else None
                        for method_name in methods}
        self.result = ImpactCoverageResult(list(methods.keys()))

    def run_batch(self, samples, labels, attrs_dict=None):
        attacked_samples, patch_mask, targets = _apply_patches(samples, labels,
                                                               self.model, self.patch_folder)
        for method_name in self.methods:
            method = self.methods[method_name]
            iou = _compute_coverage(attacked_samples, method, patch_mask,
                                    targets, writer=self._get_writer(method_name)).reshape(-1, 1)
            self.result.append(method_name, iou)
