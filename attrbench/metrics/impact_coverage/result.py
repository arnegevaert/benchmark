from typing import Tuple, Optional
from typing_extensions import override
import pandas as pd
from attrbench.metrics.result import GroupedMetricResult


class ImpactCoverageResult(GroupedMetricResult):
    def __init__(self, method_names: Tuple[str], shape: Tuple[int, ...]):
        levels = {"method": method_names}
        level_order = ("method",)
        super().__init__(method_names, shape, levels, level_order)

    @classmethod
    @override
    def _load(cls, path: str, format="hdf5") -> "ImpactCoverageResult":
        tree = cls._load_tree(path, format)
        res = ImpactCoverageResult(tree.levels["method"], tree.shape)
        res.tree = tree
        return res

    def get_df(
        self, methods: Optional[Tuple[str]] = None
    ) -> Tuple[pd.DataFrame, bool]:
        methods = methods if methods is not None else self.method_names
        df_dict = {}
        for method in methods:
            df_dict[method] = self.tree.get(method=method)
        return pd.DataFrame.from_dict(df_dict), False
