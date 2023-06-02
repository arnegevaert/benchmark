from abc import abstractmethod
from typing import Tuple, Dict
import h5py
import numpy as np
from attrbench.data.nd_array_tree.random_access_nd_array_tree import (
    RandomAccessNDArrayTree,
)
from attrbench.metrics.result import BatchResult
from attrbench import metrics
import pandas as pd
import os
import yaml


class MetricResult:
    """
    Abstract class to represent results of distributed metrics.
    """

    def __init__(
        self,
        method_names: Tuple[str, ...],
        shape: Tuple[int, ...],
        levels: Dict[str, Tuple[str, ...]],
        level_order: Tuple[str, ...],
    ):
        self.shape = shape
        self.method_names = method_names
        self.levels = levels
        self.level_order = level_order
        self.tree = RandomAccessNDArrayTree(levels, shape)

    def add(self, batch_result: BatchResult):
        """
        Adds a BatchResult to the result object.
        """
        if batch_result.method_names is not None:
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
        else:
            raise ValueError("Invalid BatchResult: no method names available")

    def save(self, path: str, format: str) -> None:
        """
        Save the result to an HDF5 file or a directory of CSV files.
        :param path: Path to the file.
        :param format: Format to save the result in. Options: hdf5, dir.
        """
        if format == "hdf5":
            with h5py.File(path, mode="x") as fp:
                fp.attrs["type"] = self.__class__.__name__
                self.tree.save_to_hdf(fp)
        elif format == "dir":
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
        elif format == "dir":
            tree = RandomAccessNDArrayTree.load_from_dir(path)
        return tree

    @classmethod
    @abstractmethod
    def _load(cls, path: str, format="hdf5") -> "MetricResult":
        """
        Load a result from an HDF5 file or a directory of CSV files
        (abstract method).
        :param path: Path to the file.
        :param format: Format to load the result from. Options: hdf5, dir.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "MetricResult":
        """
        Detect class of result file and load the result.
        """
        # If the path is a directory, load from directory of CSV files
        if os.path.isdir(path):
            with open(os.path.join(path, "metadata.yaml")) as fp:
                metadata = yaml.safe_load(fp)
            class_name = metadata["type"]
            class_obj = getattr(metrics, class_name)
            return class_obj._load(path, format="dir")
        # Otherwise, load from HDF5 file
        else:
            with h5py.File(path, "r") as fp:
                class_name = fp.attrs["type"]
                class_obj = getattr(metrics, class_name)
                return class_obj._load(path, format="hdf5")

    @abstractmethod
    def get_df(self, *args, **kwargs) -> Tuple[pd.DataFrame, bool]:
        """
        Retrieve a dataframe from the result object for some given arguments,
        along with a boolean indicating if higher is better.
        These arguments depend on the specific metric.
        """
        raise NotImplementedError
