from __future__ import annotations
import torch
import numpy as np
import h5py
from typing import List, Union, Dict, Tuple
import pandas as pd


class MetricResult:
    inverted: bool

    def __init__(self, method_names: List[str]):
        self.method_names = method_names
        # Data contains either a list of batches or a single numpy array (if the result was loaded from HDF)
        self.data: Dict[str, Union[List, np.ndarray]] = {m_name: [] for m_name in method_names}

    def add_to_hdf(self, group: h5py.Group):
        for method_name in self.method_names:
            data = torch.cat(self.data[method_name]).numpy() if type(self.data[method_name]) == list \
                else self.data[method_name]
            ds = group.create_dataset(method_name, data=data)
            ds.attrs["inverted"] = self.inverted

    def append(self, method_name, batch):
        self.data[method_name].append(batch)

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> MetricResult:
        method_names = list(group.keys())
        result = cls(method_names)
        result.data = {m_name: np.array(group[m_name]) for m_name in method_names}
        return result

    def _aggregate(self, data):
        return data

    def get_df(self, **kwargs) -> Tuple[pd.DataFrame, bool]:
        data = self.data
        for method_name in self.method_names:
            if type(data[method_name]) == list:
                data[method_name] = torch.cat(data[method_name]).numpy()
            data[method_name] = self._aggregate(data[method_name].squeeze())
        return pd.DataFrame.from_dict(data), self.inverted
