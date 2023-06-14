from typing import Tuple, Optional
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
    def __init__(
        self,
        method_names: Tuple[str],
        maskers: Tuple[str],
        mode: str,
        shape: Tuple[int, ...],
    ):
        levels = {"method": method_names, "masker": maskers}
        level_order = ("method", "masker")
        super().__init__(method_names, shape, levels, level_order)
        self.mode = mode

    def save(self, path: str, format="hdf5"):
        super().save(path, format)

        if format == "hdf5":
            with h5py.File(path, mode="a") as fp:
                fp.attrs["mode"] = self.mode
        elif format == "dir":
            with open(os.path.join(path, "metadata.yaml"), "r") as fp:
                metadata = yaml.safe_load(fp)
            metadata["mode"] = self.mode
            with open(os.path.join(path, "metadata.yaml"), "w") as fp:
                yaml.dump(metadata, fp)

    @classmethod
    def _load_tree_mode(self, path: str, format="hdf5"):
        if format == "hdf5":
            with h5py.File(path, "r") as fp:
                tree = RandomAccessNDArrayTree.load_from_hdf(fp)
                mode = fp.attrs["mode"]
        elif format == "dir":
            with open(os.path.join(path, "metadata.yaml"), "r") as fp:
                metadata = yaml.safe_load(fp)
            tree = RandomAccessNDArrayTree.load_from_dir(path)
            mode = metadata["mode"]
        return tree, mode

    @classmethod
    @override
    def _load(cls, path: str, format="hdf5") -> "MinimalSubsetResult":
        tree, mode = cls._load_tree_mode(path, format)
        res = MinimalSubsetResult(
            tree.levels["method"],
            tree.levels["masker"],
            mode,
            tree.shape,
        )
        res.tree = tree
        return res

    @override
    def get_df(
        self, masker: str, methods: Optional[Tuple[str]] = None
    ) -> Tuple[pd.DataFrame, bool]:
        higher_is_better = False
        methods = methods if methods is not None else self.method_names
        df_dict = {
            method: self.tree.get(masker=masker, method=method).flatten()
            for method in methods
        }
        return pd.DataFrame.from_dict(df_dict), higher_is_better
