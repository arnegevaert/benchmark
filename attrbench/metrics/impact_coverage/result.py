from typing import Tuple, Optional
from typing_extensions import override
import h5py
import pandas as pd

from attrbench.data.nd_array_tree.random_access_nd_array_tree import RandomAccessNDArrayTree
from attrbench.metrics.result import GroupedMetricResult
from attrbench.metrics.result.metric_result import MetricResult


class ImpactCoverageResult(GroupedMetricResult):
    def __init__(self, method_names: Tuple[str], shape: Tuple[int, ...]):
        levels = {"method": method_names}
        level_order = ("method",)
        super().__init__(method_names, shape, levels, level_order)

    @classmethod
    @override
    def load(cls, path: str) -> "MetricResult":
        with h5py.File(path, "r") as fp:
            tree = RandomAccessNDArrayTree.load_from_hdf(fp)
        res = ImpactCoverageResult(tree.levels["method"], tree.shape)
        res.tree = tree
        return res

    def get_df(self, methods: Optional[Tuple[str]] = None)\
            -> Tuple[pd.DataFrame, bool]:
        methods = methods if methods is not None else self.method_names
        df_dict = {}
        for method in methods:
            df_dict[method] = self._tree.get(method=method)
        return pd.DataFrame.from_dict(df_dict), False
