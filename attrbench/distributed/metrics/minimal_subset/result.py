from typing import Tuple, List

import h5py
import pandas as pd

from attrbench.distributed.metrics.result import BatchResult, MetricResult
from attrbench.data import RandomAccessNDArrayTree


class MinimalSubsetResult(MetricResult):
    def __init__(self, method_names: Tuple[str], maskers: Tuple[str], mode: str, shape: Tuple[int, ...]):
        super().__init__(method_names, shape)
        self.shape = shape
        self.mode = mode
        self.maskers = maskers
        self.method_names = method_names

        levels = {"method": method_names, "masker": maskers}
        self._tree = RandomAccessNDArrayTree(levels, shape)

    def add(self, batch_result: BatchResult):
        # masker -> [batch_size, 1]
        data = batch_result.results
        for method_name in set(batch_result.method_names):
            method_indices = [i for i, name in enumerate(batch_result.method_names) if name == method_name]
            for masker_name in data.keys():
                self._tree.write(
                    batch_result.indices[method_indices],
                    data[masker_name][method_indices],
                    method=method_name, masker=masker_name
                )

    def save(self, path: str):
        with h5py.File(path, mode="w") as fp:
            fp.attrs["mode"] = self.mode
            self._tree.add_to_hdf(fp)

    @classmethod
    def load(cls, path: str) -> "MinimalSubsetResult":
        with h5py.File(path, "r") as fp:
            tree = RandomAccessNDArrayTree.load_from_hdf(fp)
            res = MinimalSubsetResult(tree.levels["method"], tree.levels["masker"], fp.attrs["mode"], tree.shape)
            res._tree = tree
        return res

    def get_df(self, masker: str, methods: List[str] = None) -> Tuple[pd.DataFrame, bool]:
        higher_is_better = False
        methods = methods if methods is not None else self.method_names
        df_dict = {method: self._tree.get(masker=masker, method=method) for method in methods}
        return pd.DataFrame.from_dict(df_dict), higher_is_better
