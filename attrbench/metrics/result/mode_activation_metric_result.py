from __future__ import annotations
import torch
import numpy as np
import h5py
from typing import List, Dict, Tuple
import pandas as pd
from attrbench.metrics.result import ActivationMetricResult


class ModeActivationMetricResult(ActivationMetricResult):
    inverted: Dict[str, bool]

    def __init__(self, method_names: List[str], modes: Tuple[str], activation_fns: Tuple[str]):
        super().__init__(method_names, activation_fns)
        self.modes = modes
        self.data = {m_name: {mode: {afn: [] for afn in activation_fns} for mode in modes}
                     for m_name in self.method_names}

    def append(self, method_name, batch: Dict):
        for mode in batch.keys():
            for afn in batch[mode].keys():
                self.data[method_name][mode][afn].append(batch[mode][afn])

    def add_to_hdf(self, group: h5py.Group):
        for method_name in self.method_names:
            method_group = group.create_group(method_name)
            for mode in self.modes:
                mode_group = method_group.create_group(mode)
                for afn in self.activation_fns:
                    data = torch.cat(self.data[method_name][mode][afn]).numpy() \
                        if type(self.data[method_name][mode][afn]) == list else self.data[method_name][mode][afn]
                    ds = mode_group.create_dataset(afn, data=data)
                    ds.attrs["inverted"] = self.inverted[mode]

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> ModeActivationMetricResult:
        method_names = list(group.keys())
        modes = tuple(group[method_names[0]].keys())
        activation_fn = tuple(group[method_names[0]][modes[0]].keys())
        result = cls(method_names, modes, activation_fn)
        result.data = {
            m_name:
                {mode:
                     {afn: np.array(group[m_name][mode][afn]) for afn in activation_fn}
                 for mode in modes}
            for m_name in method_names
        }
        return result

    def get_df(self, *, mode=None, activation=None) -> Tuple[pd.DataFrame, bool]:
        data = {m_name: self._aggregate(self.data[m_name][mode][activation].squeeze())
                for m_name in self.method_names}
        df = pd.DataFrame.from_dict(data)
        return df, self.inverted[mode]
