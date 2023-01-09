from typing import Self, Tuple, Optional
import h5py
import pandas as pd

from attrbench.data import RandomAccessNDArrayTree
from attrbench.metrics.result import GroupedMetricResult


class MaxSensitivityResult(GroupedMetricResult):
    def __init__(self, method_names: Tuple[str], shape: Tuple[int, ...]):
        levels = {"method": method_names}
        level_order = ("method",)
        super().__init__(method_names, shape, levels, level_order)

    @classmethod
    def load(cls, path: str) -> Self:
        with h5py.File(path, "r") as fp:
            tree = RandomAccessNDArrayTree.load_from_hdf(fp)
        res = MaxSensitivityResult(tree.levels["method"], tree.shape)
        res.tree = tree
        return res

    def get_df(self, methods: Optional[Tuple[str]] = None)\
            -> Tuple[pd.DataFrame, bool]:
        methods = methods if methods is not None else self.method_names
        df_dict = {}
        for method in methods:
            df_dict[method] = self.tree.get(method=method)
        return pd.DataFrame.from_dict(df_dict), False
