from typing import Tuple, Optional, List
import os
import yaml
from typing_extensions import override

import h5py
import pandas as pd

from ._metric_result import MetricResult
from attribench.data.nd_array_tree._random_access_nd_array_tree import (
    RandomAccessNDArrayTree,
)


class MinimalSubsetResult(MetricResult):
    """Represents results from running the MinimalSubset metric."""

    def __init__(
        self,
        method_names: List[str],
        maskers: List[str],
        mode: str,
        num_samples: int,
    ):
        """
        Parameters
        ----------
        method_names : List[str]
            Names of attribution methods tested by MinimalSubset.
        maskers : List[str]
            Names of maskers used by MinimalSubset.
        mode : str
            Indicates if Minimal Subset Deletion or Minimal Subset Insertion
            was used.
            Options: "deletion", "insertion"
        num_samples : int
            Number of samples on which MinimalSubset was run.
        """
        levels = {"method": method_names, "masker": maskers}
        level_order = ["method", "masker"]
        shape = [num_samples, 1]
        super().__init__(method_names, shape, levels, level_order)
        self.mode = mode

    def save(self, path: str, format="hdf5"):
        super().save(path, format)

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
    def _load(cls, path: str, format="hdf5") -> "MinimalSubsetResult":
        tree, mode = cls._load_tree_mode(path, format)
        res = MinimalSubsetResult(
            tree.levels["method"],
            tree.levels["masker"],
            mode,
            tree.shape[0],
        )
        res.tree = tree
        return res

    @override
    def get_df(
        self, masker: str, methods: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, bool]:
        """Retrieves a dataframe from the result for the given masker. The
        dataframe contains a row for each method and a column for each sample.
        Each value is the MinimalSubset for the given method on the given
        sample.

        Parameters
        ----------
        masker : str
            The masker to include.
        methods : Optional[List[str]], optional
            The methods to include. If None, includes all methods.
            Defaults to None.
        """
        higher_is_better = False
        methods = methods if methods is not None else self.method_names
        df_dict = {
            method: self.tree.get(masker=masker, method=method).flatten()
            for method in methods
        }
        return pd.DataFrame.from_dict(df_dict), higher_is_better

    @override
    def merge(self, other: MetricResult, level: str, allow_overwrite: bool) -> None:
        assert isinstance(other, MinimalSubsetResult)
        if self.mode != other.mode:
            raise ValueError(
                f"Cannot merge: mode does not match: {self.mode}, {other.mode}"
            )
        super().merge(other, level, allow_overwrite)