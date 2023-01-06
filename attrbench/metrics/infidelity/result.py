from typing import Tuple, Optional, Dict
import pandas as pd
from attrbench.metrics.result import MetricResult, BatchResult
from attrbench.data import RandomAccessNDArrayTree
from numpy import typing as npt
import h5py


class InfidelityResult(MetricResult):
    def __init__(self, method_names: Tuple[str], perturbation_generators: Tuple[str],
                 activation_fns: Tuple[str], shape: Tuple[int, ...]):
        super().__init__(method_names, shape)
        self.activation_fns = activation_fns
        self.perturbation_generators = perturbation_generators

        levels = {"method": method_names,
                  "perturbation_generator": perturbation_generators,
                  "activation_fn": activation_fns}
        self._tree = RandomAccessNDArrayTree(levels, shape)

    def add(self, batch_result: BatchResult):
        # method -> perturbation_generator -> activation_fn -> [batch_size, 1]
        data: Dict[str, Dict[str, Dict[str, npt.NDArray]]] = batch_result.results
        indices = batch_result.indices.detach().cpu().numpy()
        for method_name in self.method_names:
            for perturbation_generator in self.perturbation_generators:
                for activation_fn in self.activation_fns:
                    self._tree.write(indices,
                                     data[method_name][perturbation_generator][activation_fn],
                                     method=method_name,
                                     perturbation_generator=perturbation_generator,
                                     activation_fn=activation_fn)

    def save(self, path: str):
        with h5py.File(path, mode="w") as fp:
            self._tree.add_to_hdf(fp)

    @classmethod
    def load(cls, path: str) -> "InfidelityResult":
        with h5py.File(path, "r") as fp:
            tree = RandomAccessNDArrayTree.load_from_hdf(fp)
            res = InfidelityResult(tree.levels["method"],
                                   tree.levels["perturbation_generator"],
                                   tree.levels["activation_fn"], tree.shape)
            res._tree = tree
        return res

    def get_df(self, perturbation_generator: str, activation_fn: str,
               methods: Optional[Tuple[str]] = None)\
                       -> Tuple[pd.DataFrame, bool]:
        methods = methods if methods is not None else self.method_names
        df_dict = {}
        for method in methods:
            df_dict[method] = self._tree.get(method=method,
                                             perturbation_generator=perturbation_generator,
                                             activation_fn=activation_fn)
        return pd.DataFrame.from_dict(df_dict), False
