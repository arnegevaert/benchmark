from typing import Tuple, List
import numpy as np
import pandas as pd
from attrbench.distributed.metrics.result import MetricResult, BatchResult
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
        pass

    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str) -> "InfidelityResult":
        pass

    def get_df(self, perturbation_generator: str, activation_fn: str,
               methods: Tuple[str] = None) -> Tuple[pd.DataFrame, bool]:
        methods = methods if methods is not None else self.method_names
        df_dict = {}
        for method in methods:
            df_dict[method] = self._tree.get(method=method,
                                             perturbation_generator=perturbation_generator,
                                             activation_fn=activation_fn)
        return pd.DataFrame.from_dict(df_dict), False
