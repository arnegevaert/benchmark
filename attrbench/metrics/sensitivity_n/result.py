from typing import List

import h5py
import numpy as np

from attrbench.metrics import MaskerActivationMetricResult


class SensitivityNResult(MaskerActivationMetricResult):
    inverted = False

    def __init__(self, method_names: List[str], maskers: List[str], activation_fns: List[str], index: np.ndarray):
        super().__init__(method_names, maskers, activation_fns)
        self.index = index

    def add_to_hdf(self, group: h5py.Group):
        group.attrs["index"] = self.index
        super().add_to_hdf(group)

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> MaskerActivationMetricResult:
        maskers = list(group.keys())
        activation_fns = list(group[maskers[0]].keys())
        method_names = list(group[maskers[0]][activation_fns[0]].keys())
        result = cls(method_names, maskers, activation_fns, group.attrs["index"])
        result.data = {
            masker: {
                afn: {
                    m_name: np.array(group[masker][afn][m_name]) for m_name in method_names
                } for afn in activation_fns} for masker in maskers}
        result.baseline_data = {masker: {afn: np.array(group[masker][afn]["_BASELINE"]) for afn in activation_fns}
                                for masker in maskers}
        return result

    def _aggregate(self, data):
        return np.mean(data, axis=1)


class SegSensitivityNResult(SensitivityNResult):
    pass
