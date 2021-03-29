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

    def to_df(self) -> Tuple[pd.DataFrame, bool]:
        data = self.data
        for method_name in self.method_names:
            if type(data[method_name]) == list:
                data[method_name] = torch.cat(data[method_name]).numpy()
            data[method_name] = self._aggregate(data[method_name].squeeze())
        return pd.DataFrame.from_dict(data), self.inverted


class ModeActivationMetricResult(MetricResult):
    inverted: Dict[str, bool]

    def __init__(self, method_names: List[str], modes: Tuple[str], activation_fn: Tuple[str],
                 ):
        super().__init__(method_names)
        self.modes = modes
        self.activation_fn = activation_fn
        self.data = {m_name: {mode: {afn: [] for afn in activation_fn} for mode in modes}
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
                for afn in self.activation_fn:
                    data = torch.cat(self.data[method_name][mode][afn]).numpy() \
                        if type(self.data[method_name][mode][afn]) == list else self.data[method_name][mode][afn]
                    ds = mode_group.create_dataset(afn, data=data)
                    ds.attrs["inverted"] = self.inverted[mode]

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> MetricResult:
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

    def to_df(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, bool]]:
        result = {}
        inverted = {}
        for mode in self.modes:
            for afn in self.activation_fn:
                data = {m_name: self._aggregate(self.data[m_name][mode][afn].squeeze())
                        for m_name in self.method_names}
                df = pd.DataFrame.from_dict(data)
                result[f"{mode}_{afn}"] = df
                inverted[f"{mode}_{afn}"] = self.inverted[mode]
        return result, inverted
