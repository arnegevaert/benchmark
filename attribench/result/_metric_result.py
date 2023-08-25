from abc import abstractmethod
from typing import Tuple, Dict, List
import h5py
import numpy as np
from attribench.data.nd_array_tree._random_access_nd_array_tree import (
    RandomAccessNDArrayTree,
)
from ._batch_result import BatchResult
from attribench import result
import pandas as pd
import os
import yaml


class MetricResult:
    """Abstract class to represent results of distributed metrics."""

    def __init__(
        self,
        method_names: List[str],
        shape: List[int],
        levels: Dict[str, List[str]],
        level_order: List[str],
    ):
        """
        Parameters
        ----------
        method_names : Tuple[str, ...] | List[str]
            Names of attribution methods tested by the metric.
        shape : Tuple[int, ...] | List[int]
            Shape of numpy arrays that contain the results.
            Note that this is the result on the full dataset, not a single
            sample. For example, if the metric is computed on 100 samples and
            the metric returns 10 values per sample, then the shape
            should be ``(100, 10)``.
        levels : Dict[str, Tuple[str, ...] | List[str]]
            Dictionary mapping level names to level values.
            For example::

                {
                    "method": ("a", "b"),
                    "activation_fn": ("relu", "sigmoid")
                }

        level_order : Tuple[str, ...] | List[str]
            Order of the levels in the result tree. This should contain all
            the keys in ``levels``.
        """
        # TODO method_names should be added as the final level here, rather
        # than in the subclass.
        # This would improve uniformity between subclasses.

        self.shape = shape
        self.method_names = method_names
        self.levels = levels
        self.level_order = level_order
        self.tree = RandomAccessNDArrayTree(levels, shape)

    def add(self, batch_result: BatchResult):
        """
        Adds a BatchResult to the result object.

        Parameters
        ----------
        batch_result : BatchResult
            BatchResult to add to the result object.
        """
        indices_dict = {
            method_name: np.array(
                [
                    i
                    for i, name in enumerate(batch_result.method_names)
                    if name == method_name
                ]
            )
            for method_name in set(batch_result.method_names)
        }
        target_indices = batch_result.indices.detach().cpu().numpy()
        level_order = list(self.level_order)
        level_order.remove("method")
        self.tree.write_dict_split(
            indices_dict,
            target_indices=target_indices,
            split_level="method",
            data=batch_result.results,
            level_order=level_order,
        )

    def save(self, path: str, format: str) -> None:
        """
        Save the result to an HDF5 file or a nested directory of CSV files.

        Parameters
        ----------
        path : str
            Path to the file.
        format : str
            Format to save the result in. Options: hdf5, dir.
            If hdf5, the full result is stored in a single HDF5 file.
            If csv, the result is stored in a nested directory of CSV files.
        """
        if format == "hdf5":
            with h5py.File(path, mode="x") as fp:
                fp.attrs["type"] = self.__class__.__name__
                self.tree.save_to_hdf(fp)
        elif format == "csv":
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "metadata.yaml"), "w") as fp:
                yaml.dump({"type": self.__class__.__name__}, fp)
            self.tree.save_to_dir(path)
        else:
            raise ValueError("Invalid format: {}".format(format))

    @classmethod
    def _load_tree(cls, path: str, format="hdf5") -> RandomAccessNDArrayTree:
        if format == "hdf5":
            with h5py.File(path, "r") as fp:
                tree = RandomAccessNDArrayTree.load_from_hdf(fp)
        elif format == "csv":
            tree = RandomAccessNDArrayTree.load_from_dir(path)
        else:
            raise ValueError("Invalid format", format)
        return tree

    @classmethod
    def load(cls, path: str) -> "MetricResult":
        """
        Load a result from an HDF5 file or a directory of CSV files.
        The format is inferred from the path: if the path is a directory,
        the result is loaded from a directory of CSV files, otherwise
        the result is loaded from an HDF5 file.

        The specific subclass of MetricResult is inferred from the metadata
        stored in the file or directory, and the appropriate load method
        is called.

        Parameters
        ----------
        path : str
            Path to the file or directory.

        Returns
        -------
        MetricResult
            The loaded result.
        """
        # If the path is a directory, load from directory of CSV files
        if os.path.isdir(path):
            with open(os.path.join(path, "metadata.yaml")) as fp:
                metadata = yaml.safe_load(fp)
            class_name = metadata["type"]
            class_obj = getattr(result, class_name)
            return class_obj._load(path, format="csv")
        # Otherwise, load from HDF5 file
        else:
            with h5py.File(path, "r") as fp:
                class_name = fp.attrs["type"]
                if not isinstance(class_name, str):
                    raise ValueError("Invalid type in HDF5 file")
                class_obj = getattr(result, class_name)
                return class_obj._load(path, format="hdf5")

    @abstractmethod
    def get_df(self, *args, **kwargs) -> Tuple[pd.DataFrame, bool]:
        """
        Retrieve a dataframe from the result object for some given arguments,
        along with a boolean indicating if higher is better.
        These arguments depend on the specific metric.
        """
        raise NotImplementedError

    def merge(
        self, other: "MetricResult", level: str, allow_overwrite: bool
    ) -> None:
        # Types must be equal
        if type(other) != type(self):
            raise ValueError(
                f"Cannot merge: other is of type {type(other)},"
                f" expected {type(self)}"
            )

        # Entries for the given merge level must be disjoint in the two results
        intersection = set(self.levels[level]) & set(other.levels[level])
        if len(intersection) > 0 and not allow_overwrite:
            raise ValueError(
                f"Cannot merge: level {level} has overlapping entries:"
                f" {intersection}. To merge, set allow_overwrite=True."
            )

        # Entries for all levels except the given merge level must be equal
        for l in self.levels:
            if l != level and self.levels[l] != other.levels[l]:
                raise ValueError(
                    f"Cannot merge: level {l} has different entries:"
                    f" {self.levels[l]}, {other.levels[l]}"
                )

        # Shapes must be equal
        if self.shape != other.shape:
            raise ValueError(
                f"Cannot merge: shapes do not match: {self.shape}, {other.shape}"
            )

        # Level orders must be equal
        if self.level_order != other.level_order:
            raise ValueError(
                f"Cannot merge: level orders do not match: {self.level_order},"
                f" {other.level_order}"
            )

        # If results are being merged on method level, method_names must be updated.
        # TODO this will become unnecessary once method_names is added as a
        # level in the constructor.
        if level == "method":
            self.method_names += other.method_names

        # Merge the tree and add the levels
        self.tree.merge(other.tree, level, allow_overwrite)
        self.levels[level] += other.levels[level]
        # Remove duplicates from level entries (in case of overlap)
        self.levels[level] = list(set(self.levels[level]))

    @classmethod
    @abstractmethod
    def _load(cls, path: str, format="hdf5") -> "MetricResult":
        raise NotImplementedError
