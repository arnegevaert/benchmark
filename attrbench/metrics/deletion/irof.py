from typing import Callable, List, Union, Dict

import numpy as np
import torch

from attrbench.masking import ImageMasker
from os import path
from attrbench.metrics import MaskerMetric
from ._dataset import _IrofDataset
from ._get_predictions import _get_predictions
from .result import IrofResult


def irof(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
         masker: ImageMasker, activation_fns: Union[List[str], str] = "linear",
         mode: str = "morf", start: float = 0., stop: float = 1., num_steps: int = 100):
    masking_dataset = _IrofDataset(mode, start, stop, num_steps, samples, masker)
    if type(activation_fns) == str:
        activation_fns = [activation_fns]
    masking_dataset.set_attrs(attrs)
    return _get_predictions(masking_dataset, labels, model, activation_fns)


class Irof(MaskerMetric):
    def __init__(self, model: Callable, method_names: List[str], maskers: Dict,
                 activation_fns: Union[List[str], str],
                 start: float = 0., stop: float = 1., num_steps: int = 100,
                 mode: str = "morf"):
        super().__init__(model, method_names, maskers)
        self.start = start
        self.stop = stop
        self.num_steps = num_steps
        self.mode = mode
        self.activation_fns = [activation_fns] if type(activation_fns) == str else activation_fns
        self._result = IrofResult(method_names + ["_BASELINE"], list(self.maskers.keys()),
                                  self.activation_fns, mode)

    def run_batch(self, samples, labels, attrs_dict: dict, baseline_attrs: np.ndarray):
        masking_datasets = {}
        methods_result = {masker_name: {afn: {} for afn in self.activation_fns} for masker_name in self.maskers}
        baseline_result = {masker_name: {afn: [] for afn in self.activation_fns} for masker_name in self.maskers}
        for masker_name, masker in self.maskers.items():
            masking_datasets[masker_name] = _IrofDataset(self.mode, self.start, self.stop, self.num_steps,
                                                         samples, masker)
        for masker_name, masking_dataset in masking_datasets.items():
            for method_name in attrs_dict:
                masking_dataset.set_attrs(attrs_dict[method_name])
                result = _get_predictions(masking_dataset, labels, self.model, self.activation_fns, )
                for afn in self.activation_fns:
                    methods_result[masker_name][afn][method_name] = result[afn].cpu().detach().numpy()

            for i in range(baseline_attrs.shape[0]):
                masking_dataset.set_attrs(baseline_attrs[i, ...])
                bl_result = _get_predictions(masking_dataset, labels, self.model, self.activation_fns)
                for afn in self.activation_fns:
                    baseline_result[masker_name][afn].append(bl_result[afn].cpu().detach().numpy())
            for afn in self.activation_fns:
                baseline_result[masker_name][afn] = np.stack(baseline_result[masker_name][afn], axis=1)
        self.result.append(methods_result)
        self.result.append(baseline_result, method="_BASELINE")
