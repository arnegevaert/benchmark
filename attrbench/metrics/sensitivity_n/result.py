from typing import List, Tuple

import h5py
import numpy as np

from attrbench.metrics import MetricResult, ActivationMetricResult


class SensitivityNResult(ActivationMetricResult):
    inverted = False

    def __init__(self, method_names: List[str], activation_fns: Tuple[str], index: np.ndarray):
        super().__init__(method_names, activation_fns)
        self.index = index

    def add_to_hdf(self, group: h5py.Group):
        group.attrs["index"] = self.index
        super().add_to_hdf(group)

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> MetricResult:
        method_names = list(group.keys())
        activation_fns = tuple(group[method_names[0]].keys())
        result = cls(method_names, activation_fns, group.attrs["index"])
        result.data = {m_name: {fn: np.array(group[m_name][fn]) for fn in activation_fns}
                       for m_name in method_names}
        return result

    def _aggregate(self, data):
        return np.mean(data, axis=1)


class SegSensitivityNResult(SensitivityNResult):
    pass
