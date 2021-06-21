from attrbench.metrics import MaskerActivationMetricResult
import pandas as pd
from typing import Tuple, List
import numpy as np
import h5py
from attrbench.lib import NDArrayTree


def _aoc(x: np.ndarray, columns: np.ndarray = None):
    if columns is not None:
        x = x[..., columns]
    return x[..., 0] - np.trapz(x, np.linspace(0, 1, x.shape[-1]), axis=-1)


def _auc(x: np.ndarray, columns: np.ndarray = None):
    if columns is not None:
        x = x[..., columns]
    return np.trapz(x, np.linspace(0, 1, x.shape[-1]))


class DeletionResult(MaskerActivationMetricResult):
    inverted = None

    def __init__(self, method_names: List[str], maskers: List[str], activation_fns: List[str], mode: str):
        super().__init__(method_names, maskers, activation_fns)
        if mode not in ("morf", "lerf"):
            raise ValueError("Mode must be morf or lerf")
        self.mode = mode

    def add_to_hdf(self, group: h5py.Group):
        group.attrs["mode"] = self.mode
        super().add_to_hdf(group)

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> MaskerActivationMetricResult:
        maskers = list(group.keys())
        activation_fns = list(group[maskers[0]].keys())
        method_names = list(group[maskers[0]][activation_fns[0]].keys())
        mode = group.attrs["mode"]
        result = cls(method_names, maskers, activation_fns, mode)
        result._tree = NDArrayTree.load_from_hdf(["masker", "activation_fn", "method"], group)
        return result

    def get_df(self, mode="raw", include_baseline=False, masker: str = "constant",
               activation_fn: str = "linear", columns=None, agg_fn="auc") -> Tuple[pd.DataFrame, bool]:
        if agg_fn not in ("aoc", "auc"):
            raise ValueError("agg_fn must be aoc or auc")
        postproc_fn = _aoc if agg_fn == "aoc" else _auc
        df, _ = super().get_df(mode, include_baseline, masker, activation_fn,
                               postproc_fn=lambda x: postproc_fn(x, columns))
        inverted = (self.mode == "morf" and agg_fn == "auc") or (self.mode == "lerf" and agg_fn == "aoc")
        return df, inverted


class IrofResult(DeletionResult):
    pass
