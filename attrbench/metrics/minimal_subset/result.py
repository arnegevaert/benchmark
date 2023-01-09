from typing import Tuple, Optional
from typing_extensions import override

import h5py
import pandas as pd

from attrbench.metrics.result import MetricResult
from attrbench.data import RandomAccessNDArrayTree


class MinimalSubsetResult(MetricResult):
    def __init__(self, method_names: Tuple[str], maskers: Tuple[str],
                 mode: str, shape: Tuple[int, ...]):
        levels = {"method": method_names, "masker": maskers}
        level_order = ("method", "masker")
        super().__init__(method_names, shape, levels, level_order)
        self.mode = mode

    def save(self, path: str):
        super().save(path)
        with h5py.File(path, mode="w") as fp:
            fp.attrs["mode"] = self.mode

    @classmethod
    @override
    def load(cls, path: str) -> "MinimalSubsetResult":
        with h5py.File(path, "r") as fp:
            tree = RandomAccessNDArrayTree.load_from_hdf(fp)
            res = MinimalSubsetResult(tree.levels["method"],
                                      tree.levels["masker"], fp.attrs["mode"],
                                      tree.shape)
            res.tree = tree
        return res

    @override
    def get_df(self, masker: str, methods: Optional[Tuple[str]] = None) -> Tuple[pd.DataFrame, bool]:
        higher_is_better = False
        methods = methods if methods is not None else self.method_names
        df_dict = {method: self.tree.get(masker=masker, method=method) for method in methods}
        return pd.DataFrame.from_dict(df_dict), higher_is_better
