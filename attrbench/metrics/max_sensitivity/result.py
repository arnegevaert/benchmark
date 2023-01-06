from typing import Dict, Tuple, Optional
import h5py
from numpy import typing as npt
import pandas as pd

from attrbench.data.nd_array_tree.random_access_nd_array_tree import RandomAccessNDArrayTree
from attrbench.metrics.result.batch_result import BatchResult

from attrbench.metrics.result import MetricResult


# TODO code duplication! See ImpactCoverageResult
class MaxSensitivityResult(MetricResult):
    def __init__(self, method_names: Tuple[str], shape: Tuple[int, ...]):
        super().__init__(method_names, shape)
        self._tree = RandomAccessNDArrayTree({"method": method_names}, shape)

    def add(self, batch_result: BatchResult):
        data: Dict[str, npt.NDArray] = batch_result.results
        indices = batch_result.indices.detach().cpu().numpy()
        for method_name in data.keys():
            self._tree.write(indices, data[method_name], method=method_name)

    @classmethod
    def load(cls, path: str) -> "MetricResult":
        with h5py.File(path, "r") as fp:
            tree = RandomAccessNDArrayTree.load_from_hdf(fp)
        res = MaxSensitivityResult(tree.levels["method"], tree.shape)
        res._tree = tree
        return res

    def save(self, path: str):
        with h5py.File(path, "w") as fp:
            self._tree.add_to_hdf(fp)

    def get_df(self, methods: Optional[Tuple[str]] = None)\
            -> Tuple[pd.DataFrame, bool]:
        methods = methods if methods is not None else self.method_names
        df_dict = {}
        for method in methods:
            df_dict[method] = self._tree.get(method=method)
        return pd.DataFrame.from_dict(df_dict), False
