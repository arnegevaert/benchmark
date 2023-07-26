from typing_extensions import override
from numpy import typing as npt
from typing import List, Tuple, Optional
from ._deletion_result import DeletionResult
import pandas as pd


class InsertionResult(DeletionResult):
    """
    This class serves as a simple wrapper class for Insertion results.
    It is nearly identical to the DeletionResult class, except that
    the ``higher_is_better`` flag in ``get_df`` is inverted.
    """

    def get_df(
        self,
        masker: str,
        activation_fn: str,
        agg_fn="auc",
        methods: Optional[List[str]] = None,
        columns: Optional[npt.NDArray] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        df, higher_is_better = super().get_df(
            masker, activation_fn, agg_fn, methods, columns
        )
        # The only difference between Deletion and Insertion is the fact
        # that the higher_is_better flag is inverted.
        return df, not higher_is_better

    @classmethod
    @override
    def _load(cls, path: str, format="hdf5") -> "InsertionResult":
        tree, mode = cls._load_tree_mode(path, format)
        res = InsertionResult(
            tree.levels["method"],
            tree.levels["masker"],
            tree.levels["activation_fn"],
            mode,
            tree.shape[0],
            tree.shape[1],
        )
        res.tree = tree
        return res