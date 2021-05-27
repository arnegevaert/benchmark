from typing import Callable, List, Tuple, Union, Dict

import numpy as np
import torch

from attrbench.lib.masking import Masker
from attrbench.lib.attribution_writer import AttributionWriter
from os import path
from attrbench.metrics import MaskerMetric
from ._concat_results import _concat_results
from ._dataset import _IrofIiofDataset
from ._get_predictions import _get_predictions
from .result import IrofResult, IiofResult
from attrbench.metrics import MaskerActivationMetricResult


def irof(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
         masker: Masker, activation_fn: Union[Tuple[str], str] = "linear",
         writer=None):
    masking_dataset = _IrofIiofDataset("deletion", samples, attrs, masker,
                                       writer)
    return _irof(samples, labels, model, attrs, masking_dataset, activation_fn, writer)


def iiof(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
         masker: Masker, activation_fn: Union[Tuple[str], str] = "linear",
         writer=None):
    masking_dataset = _IrofIiofDataset("insertion", samples, attrs, masker,
                                       writer)
    return _iiof(samples, labels, model, attrs, masking_dataset, activation_fn, writer)


def _irof(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
          masker: Masker, activation_fn: Union[Tuple[str], str] = "linear",
          writer=None):
    if type(activation_fn) == str:
        activation_fn = (activation_fn,)
    masking_dataset = _IrofIiofDataset("deletion",samples,attrs, masker, writer)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, masking_dataset, activation_fn,
                                                              writer)
    preds = _concat_results(orig_preds, inter_preds, neutral_preds, orig_preds)

    # Calculate AUC for each sample (depends on how many segments each sample had)
    result = {}
    for fn in activation_fn:
        auc = []
        for i in range(samples.shape[0]):
            num_segments = len(np.unique(masking_dataset.segmented_images[i, ...]))
            auc.append(np.trapz(preds[fn][i, :num_segments + 1], x=np.linspace(0, 1, num_segments + 1)))
        result[fn] = torch.tensor(auc).unsqueeze(-1)
    return result


def _iiof(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray, masker: Masker,
          activation_fn: Union[Tuple[str], str] = "linear",
          writer=None):
    if type(activation_fn) == str:
        activation_fn = (activation_fn,)
    masking_dataset = _IrofIiofDataset("insertion", samples, attrs, masker, writer)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, masking_dataset, activation_fn,
                                                              writer)
    preds = _concat_results(neutral_preds, inter_preds, orig_preds, orig_preds)

    # Calculate AUC for each sample (depends on how many segments each sample had)
    result = {}
    for fn in activation_fn:
        auc = []
        for i in range(samples.shape[0]):
            num_segments = len(np.unique(masking_dataset.segmented_images[i, ...]))
            auc.append(np.trapz(preds[fn][i, :num_segments + 1], x=np.linspace(0, 1, num_segments + 1)))
        result[fn] = torch.tensor(auc).unsqueeze(-1)
    return result


class _IrofIiof(MaskerMetric):
    def __init__(self, model: Callable, method_names: List[str], maskers: Dict,
                 activation_fns: Union[Tuple[str], str],
                 result_class: Callable, mode: str, metric_fn: Callable, writer_dir: str = None):
        super().__init__(model, method_names, maskers, writer_dir)
        self.mode = mode  # "insertion" or "deletion"
        self.activation_fns = (activation_fns,) if type(activation_fns) == str else activation_fns
        self.metric_fn = metric_fn
        self._result: MaskerActivationMetricResult = result_class(method_names + ["_BASELINE"], list(self.maskers.keys()),
                                                                 self.activation_fns)
        if self.writer_dir is not None:
            for key in self.maskers:
                self.writers[key] = AttributionWriter(path.join(self.writer_dir, key))

    def run_batch(self, samples, labels, attrs_dict: dict, baseline_attrs: np.ndarray):
        masking_datasets = {}
        methods_result = {masker_name: {afn: {} for afn in self.activation_fns} for masker_name in self.maskers}
        baseline_result = {masker_name: {afn: [] for afn in self.activation_fns} for masker_name in self.maskers}
        for masker_name, masker in self.maskers.items():
        #     masking_datasets[masker_name] = _IrofIiofDataset(self.mode, samples, masker, self._get_writer(masker_name))
        # for masker_name, masking_dataset in masking_datasets.items():
            for method_name in attrs_dict:
                result = self.metric_fn(samples, labels, self.model, attrs_dict[method_name],
                                        masker, self.activation_fns,
                                        writer=self._get_writer(method_name))
                for afn in self.activation_fns:
                    methods_result[masker_name][afn][method_name] = result[afn].cpu().detach().numpy()

            for i in range(baseline_attrs.shape[0]):
                bl_result = self.metric_fn(samples, labels, self.model, baseline_attrs[i, ...],
                                           masker, self.activation_fns)
                for afn in self.activation_fns:
                    baseline_result[masker_name][afn].append(bl_result[afn].cpu().detach().numpy())
            for afn in self.activation_fns:
                baseline_result[masker_name][afn] = np.stack(baseline_result[masker_name][afn], axis=1)
        self.result.append(methods_result)
        self.result.append(baseline_result, method="_BASELINE")


class Irof(_IrofIiof):
    def __init__(self, model: Callable, method_names: List[str], maskers: Dict,
                 activation_fns: Union[Tuple[str], str], writer_dir: str = None):
        super().__init__(model, method_names, maskers, activation_fns, IrofResult, "deletion", _irof, writer_dir)


class Iiof(_IrofIiof):
    def __init__(self, model: Callable, method_names: List[str], maskers: Dict,
                 activation_fns: Union[Tuple[str], str], writer_dir: str = None):
        super().__init__(model, method_names, maskers, activation_fns, IiofResult, "insertion", _iiof, writer_dir)
