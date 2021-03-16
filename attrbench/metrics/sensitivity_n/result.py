from typing import List, Tuple, Union, Dict

import h5py
import numpy as np
import torch
import pandas as pd

from attrbench.metrics import MetricResult


class SensitivityNResult(MetricResult):
    inverted = False

    def __init__(self, method_names: List[str], activation_fns: Tuple[str], index: np.ndarray):
        super().__init__(method_names)
        self.data = {m_name: {afn: [] for afn in activation_fns} for m_name in self.method_names}
        self.activation_fns = activation_fns
        self.index = index

    def append(self, method_name, batch):
        for afn in batch.keys():
            self.data[method_name][afn].append(batch[afn])

    def add_to_hdf(self, group: h5py.Group):
        group.attrs["index"] = self.index
        for method_name in self.method_names:
            method_group = group.create_group(method_name)
            for afn in self.activation_fns:
                data = torch.cat(self.data[method_name][afn]).numpy() if type(self.data[method_name][afn]) == list \
                    else self.data[method_name][afn]
                ds = method_group.create_dataset(afn, data=data)
                ds.attrs["inverted"] = self.inverted

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> MetricResult:
        method_names = list(group.keys())
        activation_fns = tuple(group[method_names[0]].keys())
        result = cls(method_names, activation_fns, group.attrs["index"])
        result.data = {m_name: {fn: np.array(group[m_name][fn]) for fn in activation_fns}
                       for m_name in method_names}
        return result

    def to_df(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        result = {}
        for afn in self.activation_fns:
            data = {m_name: self.data[m_name][afn].tolist() for m_name in self.method_names}
            df = pd.DataFrame.from_dict(data)
            result[afn] = df
        return result


class SegSensitivityNResult(SensitivityNResult):
    pass
