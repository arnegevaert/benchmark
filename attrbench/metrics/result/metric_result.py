from __future__ import annotations
import numpy as np
import h5py
from typing import List, Dict, Tuple, Optional
import pandas as pd


class AbstractMetricResult:
    inverted: bool

    def __init__(self, method_names: List[str]):
        self.method_names = method_names

    def _aggregate(self, data: np.ndarray):
        return data

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
        # Data contains either a list of batches or a single numpy array (if the result was loaded from HDF)
        self.data: Dict[str, np.ndarray] = {m_name: None for m_name in method_names}
        self.baseline_data: Optional[np.ndarray] = None

    def add_to_hdf(self, group: h5py.Group):
        for method_name in self.method_names:
            ds = group.create_dataset(method_name, data=self.data[method_name])
            ds.attrs["inverted"] = self.inverted
        ds = group.create_dataset("_BASELINE", data=self.baseline_data)
        ds.attrs["inverted"] = self.inverted

    def append(self, method_results: Dict, baseline_results: np.ndarray):
        for method_name in method_results.keys():
            if self.data[method_name] is not None:
                self.data[method_name] = np.concatenate([self.data[method_name], method_results[method_name]], axis=0)
            else:
                self.data[method_name] = method_results[method_name]
        if self.baseline_data is not None:
            self.baseline_data = np.concatenate([self.baseline_data, baseline_results])
        else:
            self.baseline_data = baseline_results

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> BasicMetricResult:
        method_names = list(group.keys())
        result = cls(method_names)
        result.data = {m_name: np.array(group[m_name]) for m_name in method_names}
        return result

    def get_df(self, **kwargs) -> Tuple[pd.DataFrame, bool]:
        data = self.data
        for method_name in self.method_names:
            data[method_name] = self._aggregate(data[method_name].squeeze())
        return pd.DataFrame.from_dict(data), self.inverted

