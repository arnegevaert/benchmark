from __future__ import annotations
import h5py
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from attrbench.metrics.result import AbstractMetricResult
from attrbench.lib import NDArrayTree


class MaskerActivationMetricResult(AbstractMetricResult):
    inverted: bool

    def __init__(self, method_names: List[str], maskers: List[str], activation_fns: List[str]):
        super().__init__(method_names)
        self.maskers = maskers
        self.activation_fns = activation_fns
        self.tree = NDArrayTree([
            ("masker", maskers),
            ("activation_fn", activation_fns),
            ("method", method_names)
        ])

    def append(self, data: Dict, **kwargs):
        self.tree.append(data, **kwargs)

    def add_to_hdf(self, group: h5py.Group):
        self.tree.add_to_hdf(group)

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> MaskerActivationMetricResult:
        maskers = list(group.keys())
        activation_fns = list(group[maskers[0]].keys())
        method_names = list(group[maskers[0]][activation_fns[0]].keys())
        result = cls(method_names, maskers, activation_fns)
        result.tree = NDArrayTree.load_from_hdf(["masker", "activation_fn", "method"], group)
        return result

    def get_df(self, **kwargs) -> Tuple[pd.DataFrame, bool]:
        return pd.DataFrame.from_dict(self.tree.get(postproc_fn=lambda x: np.squeeze(x, axis=-1), **kwargs)), self.inverted
