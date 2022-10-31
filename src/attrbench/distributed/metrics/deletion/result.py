import numpy as np
from typing import List, Tuple, Dict
from attrbench.lib import NDArrayTree
import pandas as pd
import torch


def _aoc(x: np.ndarray, columns: np.ndarray = None):
    if columns is not None:
        x = x[..., columns]
    return x[..., 0] - _auc(x, columns)


def _auc(x: np.ndarray, columns: np.ndarray = None):
    if columns is not None:
        x = x[..., columns]
    l = x.shape[-1] if columns is None else columns.shape[0]
    return np.sum(x, axis=-1) / l


class DeletionBatchResult:
    def __init__(self, indices: torch.Tensor, results: Dict[str, Dict[str, torch.Tensor]], method_names: List[str],
                 is_baseline: List[bool]):
        self.method_names = method_names
        self.results = results
        self.indices = indices
        self.is_baseline = is_baseline


class DeletionResult:
    def __init__(self, method_names: List[str], baseline_names: List[str],
                 maskers: List[str], activation_fns: List[str], mode: str):
        self.mode = mode
        self.activation_fns = activation_fns
        self.method_names = method_names
        self.maskers = maskers

        # TODO cannot use basic NDArrayTree, we don't append blocks.
        #   instead, we need a datastructure capable of doing random writes


        self._method_tree = NDArrayTree(
            [("masker", maskers), ("activation_fn", activation_fns), ("method", method_names)])
        self._baseline_tree = NDArrayTree(
            [("masker", maskers), ("activation_fn", activation_fns), ("baseline", baseline_names)])

    def add(self):
        pass

    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str) -> "DeletionResult":
        pass

    def get_df(self, masker: str, activation_fn: str,
               agg_fn="auc", baseline=None, include_baseline=False) -> Tuple[pd.DataFrame, bool]:
        pass
