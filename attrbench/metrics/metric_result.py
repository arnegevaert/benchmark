from __future__ import annotations
import torch
import h5py
from typing import List


class MetricResult:
    def __init__(self, method_names: List[str]):
        self.inverted = False
        self.method_names = method_names
        self.data = {m_name: [] for m_name in method_names}

    def add_to_hdf(self, group: h5py.Group):
        for method_name in self.method_names:
            group.create_dataset(method_name, data=torch.cat(self.data[method_name]).numpy())

    def append(self, method_name, batch):
        raise self.data[method_name].append(batch)

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> MetricResult:
        method_names = list(group.keys())
        result = cls(method_names)
        result.data = {m_name: [group[m_name]] for m_name in method_names}
        return result
