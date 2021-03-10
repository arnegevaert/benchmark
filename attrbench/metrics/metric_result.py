from __future__ import annotations
import torch
import numpy as np
import h5py
from typing import List, Union, Dict


class MetricResult:
    def __init__(self, method_names: List[str]):
        self.method_names = method_names
        # Data contains either a list of batches or a single numpy array (if the result was loaded from HDF)
        self.data: Dict[str, Union[List, np.ndarray]] = {m_name: [] for m_name in method_names}

    def add_to_hdf(self, group: h5py.Group):
        for method_name in self.method_names:
            if type(self.data[method_name]) == list:
                group.create_dataset(method_name, data=torch.cat(self.data[method_name]).numpy())
            else:
                group.create_dataset(method_name, data=self.data[method_name])

    def append(self, method_name, batch):
        self.data[method_name].append(batch)

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> MetricResult:
        method_names = list(group.keys())
        result = cls(method_names)
        result.data = {m_name: np.array(group[m_name]) for m_name in method_names}
        return result
