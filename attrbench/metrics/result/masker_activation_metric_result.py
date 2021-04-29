from __future__ import annotations
import h5py
from typing import List, Tuple, Dict
import pandas as pd
from attrbench.metrics.result import AbstractMetricResult
from attrbench.lib import NDArrayTree


class MaskerActivationMetricResult(AbstractMetricResult):
    inverted: bool

    def __init__(self, method_names: List[str], maskers: List[str], activation_fns: List[str]):
        super().__init__(method_names)
        self.maskers = maskers
        self.activation_fns = activation_fns
        self.tree = NDArrayTree([
            ("masker", maskers),
            ("activation_fn", activation_fns),
            ("method", method_names)
        ])

    def append(self, data: Dict):
        self.tree.append(data)

    def add_to_hdf(self, group: h5py.Group):
        for masker in self.maskers:
            masker_group = group.create_group(masker)
            for afn in self.activation_fns:
                afn_group = masker_group.create_group(afn)
                for method_name in self.method_names:
                    ds = afn_group.create_dataset(method_name, data=self.tree.get(masker=masker, activation_fn=afn,
                                                                                  method=method_name))
                    ds.attrs["inverted"] = self.inverted

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> MaskerActivationMetricResult:
        maskers = list(group.keys())
        activation_fns = list(group[maskers[0]].keys())
        method_names = list(group[maskers[0]][activation_fns[0]].keys())
        result = cls(method_names, maskers, activation_fns)
        result.append(dict(group))
        return result

    def get_df(self, **kwargs) -> Tuple[pd.DataFrame, bool]:
        return pd.DataFrame.from_dict(self.tree.get(**kwargs)), self.inverted
