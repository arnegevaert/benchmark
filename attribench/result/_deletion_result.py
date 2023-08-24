from typing_extensions import override
import os
import yaml
import h5py
import numpy as np
from numpy import typing as npt
from typing import List, Tuple, Optional
from attribench.data.nd_array_tree._random_access_nd_array_tree import (
    RandomAccessNDArrayTree,
)
from attribench.result._metric_result import MetricResult
from ._metric_result import MetricResult
import pandas as pd


def _aoc(x: np.ndarray, columns: Optional[npt.NDArray] = None):
    if columns is not None:
        x = x[..., columns]
    return x[..., 0] - _auc(x, columns)


def _auc(x: np.ndarray, columns: Optional[npt.NDArray] = None):
    # TODO do we have to normalize by the first value? Same for AOC.
    # Should actually not make a difference.
    if columns is not None:
        x = x[..., columns]
    l = x.shape[-1] if columns is None else columns.shape[0]
    return np.sum(x, axis=-1) / l


class DeletionResult(MetricResult):
    """
    Represents results from running the Deletion metric.
    """

    def __init__(
        self,
        method_names: List[str],
        maskers: List[str],
        activation_fns: List[str],
        mode: str,
        num_samples: int,
        num_steps: int,
    ):
        """
        Parameters
        ----------
        method_names : List[str]
            Names of attribution methods tested by Deletion.
        maskers : List[str]
            Names of maskers used by Deletion.
        activation_fns : List[str]
            Names of activation functions used by Deletion.
        mode : str
            Indicates if Deletion-MoRF or Deletion-LeRF was used.
            Options: "morf", "lerf"
        num_samples : int
            Number of samples on which Deletion was run.
        num_steps : int
            Number of steps used by Deletion.
        """
        levels = {
            "method": method_names,
            "masker": maskers,
            "activation_fn": activation_fns,
        }
        level_order = ["method", "masker", "activation_fn"]
        shape = [num_samples, num_steps]
        super().__init__(method_names, shape, levels, level_order)
        self.mode = mode

    @override
    def save(self, path: str, format="hdf5"):
        super().save(path, format)

        # Save additional metadata
        if format == "hdf5":
            with h5py.File(path, mode="a") as fp:
                fp.attrs["mode"] = self.mode
        elif format == "csv":
            with open(os.path.join(path, "metadata.yaml"), "r") as fp:
                metadata = yaml.safe_load(fp)
            metadata["mode"] = self.mode
            with open(os.path.join(path, "metadata.yaml"), "w") as fp:
                yaml.dump(metadata, fp)

    @classmethod
    def _load_tree_mode(
        cls, path: str, format="hdf5"
    ) -> Tuple[RandomAccessNDArrayTree, str]:
        """Loads the tree and mode from a file or directory.

        Parameters
        ----------
        path : str
            Path to the file or directory.
        format : str, optional
            Format of the saved result.
            Options: "hdf5", "csv".
            By default "hdf5".

        Returns
        -------
        Tuple[RandomAccessNDArrayTree, str]
            The RandomAccessNDArrayTree object and the mode as a string.

        Raises
        ------
        ValueError
            If the format argument is not valid.
        """
        if format == "hdf5":
            with h5py.File(path, "r") as fp:
                tree = RandomAccessNDArrayTree.load_from_hdf(fp)
                mode = str(fp.attrs["mode"])
        elif format == "csv":
            with open(os.path.join(path, "metadata.yaml"), "r") as fp:
                metadata = yaml.safe_load(fp)
            tree = RandomAccessNDArrayTree.load_from_dir(path)
            mode = metadata["mode"]
        else:
            raise ValueError("Invalid format", format)
        return tree, mode

    @classmethod
    @override
    def _load(cls, path: str, format="hdf5") -> "DeletionResult":
        tree, mode = cls._load_tree_mode(path, format)
        res = DeletionResult(
            tree.levels["method"],
            tree.levels["masker"],
            tree.levels["activation_fn"],
            mode,
            tree.shape[0],
            tree.shape[1],
        )
        res.tree = tree
        return res

    @override
    def get_df(
        self,
        masker: str,
        activation_fn: str,
        agg_fn="auc",
        methods: Optional[List[str]] = None,
        columns: Optional[npt.NDArray] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Retrieves a dataframe from the result for a given masker and
        activation function. The dataframe contains a row for each sample and a
        column for each method. Each value is the AUC/AOC for the given method
        on the given sample.

        Parameters
        ----------
        masker : str
            The masker to use.
        activation_fn : str
            The activation function to use.
        agg_fn : str
            Either "auc" for AUC or "aoc" for AOC.
        methods : Optional[List[str]]
            The methods to include. If None, includes all methods.
        columns : Optional[npt.NDArray]
            The columns used in the AUC/AOC calculation.
            If None, uses all columns.

        Returns
        -------
        Tuple[pd.DataFrame, bool]
            dataframe containing results,
            and boolean indicating if higher is better.
        """
        higher_is_better = (self.mode == "morf" and agg_fn == "aoc") or (
            self.mode == "lerf" and agg_fn == "auc"
        )
        methods = methods if methods is not None else self.method_names
        df_dict = {}
        agg_fns = {"auc": _auc, "aoc": _aoc}
        for method in methods:
            array = self.tree.get(
                masker=masker, activation_fn=activation_fn, method=method
            )
            df_dict[method] = agg_fns[agg_fn](array, columns)
        return pd.DataFrame.from_dict(df_dict), higher_is_better
    
    @override
    def merge(self, other: MetricResult, level: str, allow_overwrite: bool) -> None:
        assert isinstance(other, DeletionResult)
        if self.mode != other.mode:
            raise ValueError(
                f"Cannot merge: mode does not match: {self.mode}, {other.mode}"
            )
        super().merge(other, level, allow_overwrite)
