from typing import Tuple, Optional, List
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
    """Represents results from running the Sensitivity-N metric.
    """
    def __init__(
        self,
        method_names: List[str],
        maskers: List[str],
        activation_fns: List[str],
        num_samples: int,
        num_steps: int
    ):
        """
        Parameters
        ----------
        method_names : List[str]
            Names of attribution methods tested by Sensitivity-N.
        maskers : List[str]
            Names of maskers used by Sensitivity-N.
        activation_fns : List[str]
            Names of activation functions used by Sensitivity-N.
        num_samples : int
            Number of samples on which Sensitivity-N was run.
        num_steps : int
            Number of steps used by Sensitivity-N.
        """
        levels = {
            "masker": maskers,
            "activation_fn": activation_fns,
            "method": method_names,
        }
        shape = [num_samples, num_steps]
        level_order = ["masker", "activation_fn", "method"]
        super().__init__(method_names, shape, levels, level_order)

    @classmethod
    @override
    def _load(cls, path: str, format="hdf5") -> "SensitivityNResult":
        tree = cls._load_tree(path, format)
        res = SensitivityNResult(
            tree.levels["method"],
            tree.levels["masker"],
            tree.levels["activation_fn"],
            tree.shape[0],
            tree.shape[1]
        )
        res.tree = tree
        return res

    @override
    def get_df(
        self,
        masker: str,
        activation_fn: str,
        methods: Optional[List[str]] = None,
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
        methods : Optional[List[str, ...]]
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
