from __future__ import annotations
import numpy as np
import h5py
from typing import List, Dict, Tuple
import pandas as pd
from attrbench.lib import NDArrayTree


class AbstractMetricResult:
    inverted: bool

    def __init__(self, method_names: List[str]):
        self.method_names = method_names

    def add_to_hdf(self, group: h5py.Group):
        raise NotImplementedError

    def append(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> AbstractMetricResult:
        raise NotImplementedError

    def get_df(self, *args, **kwargs) -> Tuple[pd.DataFrame, bool]:
        raise NotImplementedError


class BasicMetricResult(AbstractMetricResult):
    inverted: bool

    def __init__(self, method_names: List[str]):
        super().__init__(method_names)
        self.tree = NDArrayTree([("method", self.method_names)])

    def add_to_hdf(self, group: h5py.Group):
        for method_name in self.method_names:
            ds = group.create_dataset(method_name, data=self.tree.get(methods=[method_name]))
            ds.attrs["inverted"] = self.inverted

    def append(self, data: Dict, **kwargs):
        self.tree.append(data, **kwargs)

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> BasicMetricResult:
        method_names = list(group.keys())
        result = cls(method_names)
        result.append({m_name: np.array(group[m_name]) for m_name in method_names})
        return result

    def get_df(self, **kwargs) -> Tuple[pd.DataFrame, bool]:
        return pd.DataFrame.from_dict(self.tree.get(**kwargs)), self.inverted
