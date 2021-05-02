from __future__ import annotations
import numpy as np
import h5py
from typing import List, Dict, Tuple
import pandas as pd
from attrbench.lib import NDArrayTree


class AbstractMetricResult:
    inverted: bool

    def __init__(self, method_names: List[str]):
        self.method_names = method_names

    def add_to_hdf(self, group: h5py.Group):
        raise NotImplementedError

    def append(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> AbstractMetricResult:
        raise NotImplementedError

    def get_df(self, *args, **kwargs) -> Tuple[pd.DataFrame, bool]:
        raise NotImplementedError


class BasicMetricResult(AbstractMetricResult):
    inverted: bool

    def __init__(self, method_names: List[str]):
        super().__init__(method_names)
        self.tree = NDArrayTree([("method", self.method_names)])

    def add_to_hdf(self, group: h5py.Group):
        self.tree.add_to_hdf(group)

    def append(self, data: Dict, **kwargs):
        self.tree.append(data, **kwargs)

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> BasicMetricResult:
        method_names = list(group.keys())
        result = cls(method_names)
        result.tree = NDArrayTree.load_from_hdf(["method"], group)
        return result

    def get_df(self, mode="raw") -> Tuple[pd.DataFrame, bool]:
        raw_results = pd.DataFrame.from_dict(
            self.tree.get(
                postproc_fn=lambda x: np.squeeze(x, axis=-1),
                exclude=dict(method=["_BASELINE"])
            )
        )
        if mode == "raw":
            return raw_results, self.inverted
        else:
            baseline_results = pd.DataFrame(self.tree.get(
                postproc_fn=lambda x: np.squeeze(x, axis=-1),
                select=dict(method=["_BASELINE"])
            )["_BASELINE"])
            baseline_avg = baseline_results.mean(axis=1)
            if mode == "raw_dist":
                return raw_results.sub(baseline_avg, axis=0), self.inverted
            elif mode == "std_dist":
                return raw_results\
                           .sub(baseline_avg, axis=0)\
                           .div(baseline_results.std(axis=1), axis=0),\
                       self.inverted
            else:
                raise ValueError(f"Invalid value for argument mode: {mode}. Must be raw, raw_dist or std_dist.")
