from __future__ import annotations
import torch
import numpy as np
import h5py
from typing import List, Tuple
import pandas as pd
from attrbench.metrics.result import MetricResult


class ActivationMetricResult(MetricResult):
    def __init__(self, method_names: List[str], activation_fns: Tuple[str]):
        super().__init__(method_names)
        self.data = {m_name: {afn: [] for afn in activation_fns} for m_name in self.method_names}
        self.activation_fns = activation_fns

    def append(self, method_name, batch):
        for afn in batch.keys():
            self.data[method_name][afn].append(batch[afn])

    def add_to_hdf(self, group: h5py.Group):
        for method_name in self.method_names:
            method_group = group.create_group(method_name)
            for afn in self.activation_fns:
                data = torch.cat(self.data[method_name][afn]).numpy() if type(self.data[method_name][afn]) == list \
                    else self.data[method_name][afn]
                ds = method_group.create_dataset(afn, data=data)
                ds.attrs["inverted"] = self.inverted

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> ActivationMetricResult:
        method_names = list(group.keys())
        activation_fns = tuple(group[method_names[0]].keys())
        result = cls(method_names, activation_fns)
        result.data = {m_name: {fn: np.array(group[m_name][fn]) for fn in activation_fns}
                       for m_name in method_names}
        return result

    def get_df(self, *, activation=None) -> Tuple[pd.DataFrame, bool]:
        data = {m_name: self._aggregate(self.data[m_name][activation].squeeze())
                for m_name in self.method_names}
        df = pd.DataFrame.from_dict(data)
        return df, self.inverted


