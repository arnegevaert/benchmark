from typing import Tuple, Optional
from typing_extensions import override
import pandas as pd
from ._grouped_metric_result import GroupedMetricResult


class InfidelityResult(GroupedMetricResult):
    def __init__(
        self,
        method_names: Tuple[str],
        perturbation_generators: Tuple[str],
        activation_fns: Tuple[str],
        shape: Tuple[int, ...],
    ):
        levels = {
            "method": method_names,
            "perturbation_generator": perturbation_generators,
            "activation_fn": activation_fns,
        }
        level_order = ("method", "perturbation_generator", "activation_fn")
        super().__init__(method_names, shape, levels, level_order)

    @classmethod
    @override
    def _load(cls, path: str, format="hdf5") -> "InfidelityResult":
        tree = cls._load_tree(path, format)
        res = InfidelityResult(
            tree.levels["method"],
            tree.levels["perturbation_generator"],
            tree.levels["activation_fn"],
            tree.shape,
        )
        res.tree = tree
        return res

    def get_df(
        self,
        perturbation_generator: str,
        activation_fn: str,
        methods: Optional[Tuple[str]] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        methods = methods if methods is not None else self.method_names
        df_dict = {}
        for method in methods:
            df_dict[method] = self.tree.get(
                method=method,
                perturbation_generator=perturbation_generator,
                activation_fn=activation_fn,
            ).flatten()
        return pd.DataFrame.from_dict(df_dict), False