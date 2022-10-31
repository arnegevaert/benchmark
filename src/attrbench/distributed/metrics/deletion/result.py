import h5py
import numpy as np
from numpy import typing as npt
from typing import List, Tuple, Dict
from attrbench.data import RandomAccessNDArrayTree
import pandas as pd
import torch


def _aoc(x: np.ndarray, columns: npt.NDArray = None):
    if columns is not None:
        x = x[..., columns]
    return x[..., 0] - _auc(x, columns)


def _auc(x: np.ndarray, columns: npt.NDArray = None):
    if columns is not None:
        x = x[..., columns]
    l = x.shape[-1] if columns is None else columns.shape[0]
    return np.sum(x, axis=-1) / l


class DeletionBatchResult:
    def __init__(self, indices: torch.Tensor, results: Dict[str, Dict[str, torch.Tensor]], method_names: List[str]):
        self.method_names = method_names
        self.results = results
        self.indices = indices


class DeletionResult:
    def __init__(self, method_names: List[str],
                 maskers: List[str], activation_fns: List[str], mode: str,
                 shape: Tuple[int, ...]):
        self.mode = mode
        self.activation_fns = activation_fns
        self.method_names = method_names
        self.maskers = maskers

        levels = {"masker": maskers, "activation_fn": activation_fns, "method": method_names}
        self._tree = RandomAccessNDArrayTree(levels, shape)

    def add(self, batch_result: DeletionBatchResult):
        """
        Adds a DeletionBatchResult to the result object.
        A DeletionBatchResult can contain results from multiple methods and arbitrary sample indices,
        so this method uses the random access functionality of the RandomAccessNDArrayTree to save it.
        """
        data = batch_result.results
        for masker_name in data.keys():
            for activation_fn in data[masker_name].keys():
                for method_name in set(batch_result.method_names):
                    method_indices = [i for i, name in enumerate(batch_result.method_names) if name == method_name]
                    self._tree.write(
                        batch_result.indices[method_indices],
                        data[masker_name][activation_fn][method_indices])

    def save(self, path: str):
        """
        Saves the DeletionResult to an HDF5 file.
        """
        with h5py.File(path, mode="w") as fp:
            fp.attrs["mode"] = self.mode
            self._tree.add_to_hdf(fp)

    @classmethod
    def load(cls, path: str) -> "DeletionResult":
        """
        Loads a DeletionResult from an HDF5 file.
        """
        with h5py.File(path, "r") as fp:
            tree = RandomAccessNDArrayTree.load_from_hdf(fp)
            res = DeletionResult(tree.levels["method"], tree.levels["masker"],
                                 tree.levels["activation_fn"], fp.attrs["mode"], tree.shape)
            res._tree = tree
        return res

    def get_df(self, masker: str, activation_fn: str, agg_fn="auc", methods: List[str] = None,
               columns: npt.NDArray = None) -> Tuple[pd.DataFrame, bool]:
        """
        Retrieves a dataframe from the result for a given masker and activation function.
        The dataframe contains a row for each sample and a column for each method.
        Each value is the AUC/AOC for the given method on the given sample.
        :param masker: the masker to use
        :param activation_fn: the activation function to use
        :param agg_fn: either "auc" for AUC or "aoc" for AOC
        :param methods: the methods to include. If None, includes all methods.
        :param columns: the columns used in the AUC/AOC calculation
        :return: dataframe containing results, and boolean indicating if higher is better
        """
        higher_is_better = (self.mode == "morf" and agg_fn == "aoc") or (self.mode == "lerf" and agg_fn == "auc")
        methods = methods if methods is not None else self.method_names
        df_dict = {}
        agg_fns = {"auc": _auc, "aoc": _aoc}
        for method in methods:
            array = self._tree.get(masker=masker, activation_fn=activation_fn, method=method)
            df_dict[method] = agg_fns[agg_fn](array, columns)
        return pd.DataFrame.from_dict(df_dict), higher_is_better
