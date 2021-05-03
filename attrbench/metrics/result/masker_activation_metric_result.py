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

    def _postproc_fn(self, x):
        return np.squeeze(x, axis=-1)

    def get_df(self, mode="raw", include_baseline=False, masker: str = "constant", activation_fn: str = "linear") -> Tuple[pd.DataFrame, bool]:
        raw_results = pd.DataFrame.from_dict(
            self.tree.get(
                postproc_fn=self._postproc_fn,
                exclude=dict(method=["_BASELINE"]),
                select=dict(masker=[masker], activation_fn=[activation_fn])
            )[masker][activation_fn]
        )
        baseline_results = pd.DataFrame(self.tree.get(
            postproc_fn=self._postproc_fn,
            select=dict(method=["_BASELINE"], masker=[masker], activation_fn=[activation_fn])
        )[masker][activation_fn]["_BASELINE"])
        if include_baseline:
            raw_results["Baseline"] = baseline_results.iloc[:, 0]
        if mode == "raw":
            return raw_results, self.inverted
        else:
            baseline_avg = baseline_results.mean(axis=1)
            if mode == "raw_dist":
                return raw_results.sub(baseline_avg, axis=0), self.inverted
            elif mode == "std_dist":
                return raw_results \
                           .sub(baseline_avg, axis=0) \
                           .div(baseline_results.std(axis=1), axis=0), \
                       self.inverted
            else:
                raise ValueError(f"Invalid value for argument mode: {mode}. Must be raw, raw_dist or std_dist.")
