from typing import List, Optional, Tuple
from typing_extensions import override
import pandas as pd
from ._grouped_metric_result import GroupedMetricResult


class ImpactCoverageResult(GroupedMetricResult):
    """
    Represents results from running the ImpactCoverage metric.
    """
    def __init__(self, method_names: List[str], num_samples: int):
        """
        Parameters
        ----------
        method_names : List[str]
            Names of attribution methods tested by ImpactCoverage.
        num_samples : int
            Number of samples on which ImpactCoverage was run.
        """
        levels = {"method": method_names}
        level_order = ["method"]
        super().__init__(method_names, [num_samples], levels, level_order)

    @classmethod
    @override
    def _load(cls, path: str, format="hdf5") -> "ImpactCoverageResult":
        tree = cls._load_tree(path, format)
        res = ImpactCoverageResult(tree.levels["method"], tree.shape[0])
        res.tree = tree
        return res

    def get_df(
        self, methods: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, bool]:
        """Retrieves a dataframe from the result. The dataframe contains a row
        for each method and a column for each sample. Each value is the
        ImpactCoverage for the given method on the given sample.

        Parameters
        ----------
        methods : Optional[List[str]], optional
            the methods to include. If None, includes all methods.
            Defaults to None.

        Returns
        -------
        Tuple[pd.DataFrame, bool]
            Dataframe containing results,
            and boolean indicating if higher is better.
        """
        methods = methods if methods is not None else self.method_names
        df_dict = {}
        for method in methods:
            df_dict[method] = self.tree.get(method=method)
        return pd.DataFrame.from_dict(df_dict), True
