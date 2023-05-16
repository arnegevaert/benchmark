from abc import abstractmethod
from typing import Tuple, Dict
import h5py
import numpy as np
from attrbench.data.nd_array_tree.random_access_nd_array_tree import RandomAccessNDArrayTree
from attrbench.metrics.result import BatchResult
from attrbench import metrics
import pandas as pd


class MetricResult:
    """
    Abstract class to represent results of distributed metrics.
    """
    def __init__(self, method_names: Tuple[str, ...], shape: Tuple[int, ...],
                 levels: Dict[str, Tuple[str, ...]],
                 level_order: Tuple[str, ...]):
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
                        [i for i, name in enumerate(batch_result.method_names) 
                         if name == method_name]
                        )
                    for method_name in set(batch_result.method_names)
                    }
            target_indices = batch_result.indices.detach().cpu().numpy()
            level_order = list(self.level_order)
            level_order.remove("method")
            self.tree.write_dict_split(indices_dict,
                                       target_indices=target_indices,
                                       split_level="method",
                                       data=batch_result.results,
                                       level_order=level_order)
        else:
            raise ValueError("Invalid BatchResult: no method names available")

    def save(self, path: str):
        """
        Save the result to an HDF5 file.
        """
        with h5py.File(path, mode="x") as fp:
            fp.attrs["type"] = self.__class__.__name__
            self.tree.add_to_hdf(fp)

    @classmethod
    @abstractmethod
    def _load(cls, path: str) -> "MetricResult":
        """
        Load a result from an HDF5 file (abstract method).
        """
        raise NotImplementedError
    
    @classmethod
    def load(cls, path: str) -> "MetricResult":
        """
        Detect class of result file and load the result.
        """
        with h5py.File(path, "r") as fp:
            class_name = fp.attrs["type"]
            class_obj = getattr(metrics, class_name)
            return class_obj._load(path)

    @abstractmethod
    def get_df(self, *args, **kwargs) -> Tuple[pd.DataFrame, bool]:
        """
        Retrieve a dataframe from the result object for some given arguments,
        along with a boolean indicating if higher is better.
        These arguments depend on the specific metric.
        """
        raise NotImplementedError
