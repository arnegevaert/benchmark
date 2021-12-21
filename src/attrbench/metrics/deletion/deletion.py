from typing import Callable, List, Union, Tuple, Dict

import numpy as np
import torch

from attrbench.lib import AttributionWriter
from attrbench.lib.masking import Masker
from attrbench.metrics import MaskerMetric
from ._dataset import _DeletionDataset
from .result import DeletionResult
from ._get_predictions import _get_predictions


def deletion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
             masker: Masker,
             activation_fns: Union[List[str], str] = "linear",
             mode: str = "morf", start: float = 0., stop: float = 1., num_steps: int = 100,
             writer: AttributionWriter = None) -> Dict:
    if type(activation_fns) == str:
        activation_fns = [activation_fns]
    ds = _DeletionDataset(mode, start, stop, num_steps, samples, attrs, masker)
    return _get_predictions(ds, labels, model, activation_fns, writer)


class Deletion(MaskerMetric):
    def __init__(self, model: Callable, method_names: List[str], maskers: Dict,
                 activation_fns: Union[Tuple[str], str], mode: str = "morf",
                 start: float = 0., stop: float = 1., num_steps: int = 100,
                 writer_dir: str = None):
        super().__init__(model, method_names, maskers, writer_dir)
        self.start = start
        self.stop = stop
        self.num_steps = num_steps
        self.mode = mode
        self.activation_fns = [activation_fns] if type(activation_fns) == str else list(activation_fns)
        self._result: DeletionResult = DeletionResult(method_names + ["_BASELINE"], list(self.maskers.keys()),
                                                      self.activation_fns, mode)

    def run_batch(self, samples, labels, attrs_dict: dict, baseline_attrs: np.ndarray):
        methods_result = {masker_name: {afn: {} for afn in self.activation_fns} for masker_name in self.maskers}
        baseline_result = {masker_name: {afn: [] for afn in self.activation_fns} for masker_name in self.maskers}
        for masker_name, masker in self.maskers.items():
            methods_result[masker_name] = {afn: {} for afn in self.activation_fns}
            for method_name in attrs_dict:
                result = deletion(samples, labels, self.model,
                                  attrs_dict[method_name], masker,
                                  self.activation_fns, self.mode,
                                  self.start, self.stop, self.num_steps,
                                  self._get_writer(method_name))
                for afn in self.activation_fns:
                    methods_result[masker_name][afn][method_name] = result[afn].cpu().detach().numpy()

            for i in range(baseline_attrs.shape[0]):
                bl_result = deletion(samples, labels, self.model,
                                     baseline_attrs[i, ...], masker,
                                     self.activation_fns, self.mode,
                                     self.start, self.stop, self.num_steps)
                for afn in self.activation_fns:
                    baseline_result[masker_name][afn].append(bl_result[afn].cpu().detach().numpy())
            for afn in self.activation_fns:
                baseline_result[masker_name][afn] = np.stack(baseline_result[masker_name][afn], axis=1)
        self.result.append(methods_result)
        self.result.append(baseline_result, method="_BASELINE")
