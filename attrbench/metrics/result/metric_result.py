from __future__ import annotations
import numpy as np
import h5py
from typing import List, Dict, Tuple, Optional
import pandas as pd
from attrbench.lib import NDArrayTree


class AbstractMetricResult:
    inverted: bool

    def __init__(self, method_names: List[str]):
        self.method_names = method_names
        self._tree: Optional[NDArrayTree] = None

    @property
    def tree(self) -> NDArrayTree:
        if self._tree is not None:
            return self._tree
        raise NotImplementedError

    def add_to_hdf(self, group: h5py.Group):
        self.tree.add_to_hdf(group)

    def append(self, data: Dict, **kwargs):
        self.tree.append(data, **kwargs)

    def _get_df(self, raw_results: pd.DataFrame, baseline_results: pd.DataFrame,
                mode: str, include_baseline: bool) -> Tuple[pd.DataFrame, bool]:
        if include_baseline:
            raw_results["Baseline"] = baseline_results.iloc[:, 0]
        if mode == "raw":
            return raw_results, self.inverted
        elif mode == "single_dist":
            return raw_results.sub(baseline_results.iloc[:, 0], axis=0), self.inverted
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

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> AbstractMetricResult:
        raise NotImplementedError

    def get_df(self, *args, **kwargs) -> Tuple[pd.DataFrame, bool]:
        raise NotImplementedError


class BasicMetricResult(AbstractMetricResult):
    inverted: bool

    def __init__(self, method_names: List[str]):
        super().__init__(method_names)
        self._tree = NDArrayTree([("method", self.method_names)])

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> BasicMetricResult:
        method_names = list(group.keys())
        result = cls(method_names)
        result._tree = NDArrayTree.load_from_hdf(["method"], group)
        return result

    def get_df(self, mode="raw", include_baseline=False) -> Tuple[pd.DataFrame, bool]:
        raw_results = pd.DataFrame.from_dict(
            self.tree.get(
                postproc_fn=lambda x: np.squeeze(x, axis=-1),
                exclude=dict(method=["_BASELINE"])
            )
        )
        baseline_results = pd.DataFrame(self.tree.get(
            postproc_fn=lambda x: np.squeeze(x, axis=-1),
            select=dict(method=["_BASELINE"])
        )["_BASELINE"])
        return self._get_df(raw_results, baseline_results, mode, include_baseline)
