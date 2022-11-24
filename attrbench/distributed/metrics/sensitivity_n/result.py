from typing import Tuple, List
import numpy as np
import pandas as pd
from attrbench.distributed.metrics.result import MetricResult, BatchResult
from attrbench.data import RandomAccessNDArrayTree
from numpy import typing as npt
import h5py


def _column_avg(x: npt.NDArray, columns: npt.NDArray) -> npt.NDArray:
    """
    Returns the average value of x along the specified columns.
    Used in get_df.
    """
    if columns is not None:
        x = x[..., columns]
    return np.mean(x, axis=-1)


# TODO loads of code duplication here with DeletionResult
class SensitivityNResult(MetricResult):
    def __init__(self, method_names: Tuple[str],
                 maskers: Tuple[str], activation_fns: Tuple[str],
                 shape: Tuple[int, ...]):
        super().__init__(method_names, shape)
        self.activation_fns = activation_fns
        self.maskers = maskers

        levels = {"method": method_names, "masker": maskers, "activation_fn": activation_fns}
        self._tree = RandomAccessNDArrayTree(levels, shape)

    def add(self, batch_result: BatchResult):
        """
        Adds a BatchResult to the result object.
        A BatchResult can contain results from multiple methods and arbitrary sample indices,
        so this method uses the random access functionality of the RandomAccessNDArrayTree to save it.
        """
        data = batch_result.results
        for method_name in set(batch_result.method_names):
            method_indices = [i for i, name in enumerate(batch_result.method_names) if name == method_name]
            for masker_name in data.keys():
                for activation_fn in data[masker_name].keys():
                    self._tree.write(
                        batch_result.indices[method_indices],
                        data[method_indices][masker_name][activation_fn],
                        method=method_name, masker=masker_name, activation_fn=activation_fn)

    def save(self, path: str):
        """
        Saves the SensitivityNResult to an HDF5 file.
        """
        with h5py.File(path, mode="w") as fp:
            self._tree.add_to_hdf(fp)

    @classmethod
    def load(cls, path: str) -> "SensitivityNResult":
        """
        Loads a SensitivityNResult from an HDF5 file.
        """
        with h5py.File(path, "r") as fp:
            tree = RandomAccessNDArrayTree.load_from_hdf(fp)
            res = SensitivityNResult(tree.levels["method"], tree.levels["masker"],
                                     tree.levels["activation_fn"], tree.shape)
            res._tree = tree
        return res

    def get_df(self, masker: str, activation_fn: str,
               methods: List[str], columns: npt.NDArray) -> Tuple[pd.DataFrame, bool]:
        """
        Retrieves a dataframe from the result for a given masker and activation function.
        The dataframe contains a row for each sample and a column for each method.
        Each value is the average Sensitivity-N value for the given method on the given sample,
        over the specified columns.
        :param masker: the masker to use
        :param activation_fn: the activation function to use
        :param methods: the methods to include. If None, includes all methods.
        :param columns: the columns used in the aggregation
        :return: dataframe containing results, and boolean indicating if higher is better (always True for this metric)
        """
        methods = methods if methods is not None else self.method_names
        df_dict = {}
        for method in methods:
            array = self._tree.get(masker=masker, activation_fn=activation_fn, method=method)
            df_dict[method] = _column_avg(array, columns)
        return pd.DataFrame.from_dict(df_dict), True
