from typing import Callable, List, Union, Tuple, Dict

import numpy as np
import torch

from attrbench.lib import AttributionWriter
from attrbench.lib.masking import Masker
from attrbench.metrics import Metric
from ._concat_results import _concat_results
from ._dataset import _InsertionDeletionDataset
from ._get_predictions import _get_predictions
from .result import InsertionResult, DeletionResult


def insertion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
              num_steps: int, masker: Masker, reverse_order: bool = False,
              activation_fn: Union[Tuple[str], str] = "linear",
              writer: AttributionWriter = None) -> Dict:
    if type(activation_fn) == str:
        activation_fn = (activation_fn,)
    ds = _InsertionDeletionDataset("insertion", num_steps, samples, attrs, masker, reverse_order)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, ds, activation_fn,
                                                              writer)
    return _concat_results(neutral_preds, inter_preds, orig_preds, orig_preds)


def deletion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
             num_steps: int, masker: Masker, reverse_order: bool = False,
             activation_fn: Union[Tuple[str], str] = "linear",
             writer: AttributionWriter = None) -> Dict:
    if type(activation_fn) == str:
        activation_fn = (activation_fn,)
    ds = _InsertionDeletionDataset("deletion", num_steps, samples, attrs, masker, reverse_order)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, ds, activation_fn,
                                                              writer)
    return _concat_results(orig_preds, inter_preds, neutral_preds, orig_preds)


class _InsertionDeletion(Metric):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, masker: Masker,
                 mode: Union[Tuple[str], str], activation_fn: Union[Tuple[str], str],
                 result_class: Callable, method_fn: Callable, writer_dir: str = None):
        super().__init__(model, method_names, writer_dir)
        self.num_steps = num_steps
        self.masker = masker
        self.modes = (mode,) if type(mode) == str else mode
        for m in self.modes:
            if m not in ("morf", "lerf"):
                raise ValueError(f"Invalid mode: {m}")
        self.activation_fn = (activation_fn,) if type(activation_fn) == str else activation_fn
        self.result = result_class(method_names, self.modes, self.activation_fn)
        self.method_fn = method_fn

    def run_batch(self, samples, labels, attrs_dict: dict):
        for method_name in attrs_dict:
            method_result = {}
            for mode in self.modes:
                reverse_order = mode == "lerf"
                method_result[mode] = self.method_fn(samples, labels, self.model,
                                                     attrs_dict[method_name], self.num_steps, self.masker,
                                                     reverse_order, self.activation_fn,
                                                     self._get_writer(method_name))
            self.result.append(method_name, method_result)


class Insertion(_InsertionDeletion):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, masker: Masker,
                 mode: Union[Tuple[str], str], activation_fn: Union[Tuple[str], str],
                 writer_dir: str = None):
        super().__init__(model, method_names, num_steps, masker, mode, activation_fn,
                         InsertionResult, insertion, writer_dir)


class Deletion(_InsertionDeletion):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, masker: Masker,
                 mode: Union[Tuple[str], str], activation_fn: Union[Tuple[str], str],
                 writer_dir: str = None):
        super().__init__(model, method_names, num_steps, masker, mode, activation_fn,
                         DeletionResult, deletion, writer_dir)


