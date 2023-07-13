from typing import Tuple, Optional
from typing_extensions import override
import numpy as np
import pandas as pd
from numpy import typing as npt

from ._grouped_metric_result import GroupedMetricResult


def _column_avg(
    x: npt.NDArray, columns: Optional[npt.NDArray] = None
) -> npt.NDArray:
    """
    Returns the average value of x along the specified columns.
    Used in get_df.
    """
    if columns is not None:
        x = x[..., columns]
    return np.mean(x, axis=-1)


# TODO loads of code duplication here with DeletionResult
class SensitivityNResult(GroupedMetricResult):
    def __init__(
        self,
        method_names: Tuple[str],
        maskers: Tuple[str],
        activation_fns: Tuple[str],
        shape: Tuple[int, ...],
    ):
        levels = {
            "masker": maskers,
            "activation_fn": activation_fns,
            "method": method_names,
        }
        level_order = ["method", "masker", "activation_fn"]
        super().__init__(method_names, shape, levels, level_order)

    @classmethod
    @override
    def _load(cls, path: str, format="hdf5") -> "SensitivityNResult":
        tree = cls._load_tree(path, format)
        res = SensitivityNResult(
            tree.levels["masker"],
            tree.levels["activation_fn"],
            tree.levels["method"],
            tree.shape,
        )
        res.tree = tree
        return res

    @override
    def get_df(
        self,
        masker: str,
        activation_fn: str,
        methods: Optional[Tuple[str, ...]] = None,
        columns: Optional[npt.NDArray] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Retrieves a dataframe from the result for a given masker and
        activation function. The dataframe contains a row for each sample and a
        column for each method. Each value is the average Sensitivity-N value
        for the given method on the given sample,
        over the specified columns.

        Parameters
        ----------
        masker : str
            the masker to use
        activation_fn : str
            the activation function to use
        methods : Optional[Tuple[str, ...]]
            the methods to include. If None, includes all methods.
        columns : Optional[npt.NDArray]
            the columns used in the aggregation.
            If None, uses all columns.

        Returns
        -------
        Tuple[pd.DataFrame, bool]
            dataframe containing results,
            and boolean indicating if higher is better
            (always True for Sensitivity-N)
        """

        methods = methods if methods is not None else self.method_names
        df_dict = {}
        for method in methods:
            array = self.tree.get(
                masker=masker, activation_fn=activation_fn, method=method
            )
            df_dict[method] = _column_avg(array, columns)
        return pd.DataFrame.from_dict(df_dict), True
