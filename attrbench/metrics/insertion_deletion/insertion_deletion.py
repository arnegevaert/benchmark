from typing import Callable, List, Union, Tuple, Dict

import numpy as np
import torch

from attrbench.lib import AttributionWriter
from attrbench.lib.masking import Masker
from attrbench.metrics import MaskerMetric
from ._concat_results import _concat_results
from ._dataset import _InsertionDeletionDataset
from ._get_predictions import _get_predictions
from .result import InsertionResult, DeletionResult, InsertionDeletionResult


def insertion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
              num_steps: int, masker: Masker,
              activation_fn: Union[Tuple[str], str] = "linear",
              writer: AttributionWriter = None) -> Dict:
    if type(activation_fn) == str:
        activation_fn = (activation_fn,)
    ds = _InsertionDeletionDataset("insertion", num_steps, samples, attrs, masker)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, ds, activation_fn,
                                                              writer)
    return _concat_results(neutral_preds, inter_preds, orig_preds, orig_preds)


def deletion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
             num_steps: int, masker: Masker,
             activation_fn: Union[Tuple[str], str] = "linear",
             writer: AttributionWriter = None) -> Dict:
    if type(activation_fn) == str:
        activation_fn = (activation_fn,)
    ds = _InsertionDeletionDataset("deletion", num_steps, samples, attrs, masker)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, ds, activation_fn,
                                                              writer)
    return _concat_results(orig_preds, inter_preds, neutral_preds, orig_preds)


class _InsertionDeletion(MaskerMetric):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, maskers: Dict,
                 activation_fns: Union[Tuple[str], str],
                 result_class: Callable, method_fn: Callable, writer_dir: str = None):
        super().__init__(model, method_names, maskers, writer_dir)
        self.num_steps = num_steps
        self.activation_fns = (activation_fns,) if type(activation_fns) == str else activation_fns
        self.result: InsertionDeletionResult = result_class(method_names, self.activation_fns)
        self.method_fn = method_fn

    def run_batch(self, samples, labels, attrs_dict: dict):
        for method_name in attrs_dict:
            for masker_name, masker in self.maskers.items():
                result = self.method_fn(samples, labels, self.model,
                                        attrs_dict[method_name], self.num_steps, masker,
                                        self.activation_fns,
                                        self._get_writer(method_name))
                for afn in self.activation_fns:
                    self.result.append(method_name, masker_name, afn, result[afn])


class Insertion(_InsertionDeletion):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, maskers: Dict,
                 activation_fns: Union[Tuple[str], str], writer_dir: str = None):
        super().__init__(model, method_names, num_steps, maskers, activation_fns,
                         InsertionResult, insertion, writer_dir)


class Deletion(_InsertionDeletion):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, maskers: Dict,
                 activation_fns: Union[Tuple[str], str], writer_dir: str = None):
        super().__init__(model, method_names, num_steps, maskers, activation_fns,
                         DeletionResult, deletion, writer_dir)
