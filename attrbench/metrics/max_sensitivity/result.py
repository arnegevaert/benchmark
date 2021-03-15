from typing import List

import h5py
import numpy as np

from attrbench.metrics import MetricResult


class MaxSensitivityResult(MetricResult):
    inverted = True

    def __init__(self, method_names: List[str], radius: float):
        super().__init__(method_names)
        self.inverted = True
        self.radius = radius

    def add_to_hdf(self, group: h5py.Group):
        super().add_to_hdf(group)
        group.attrs["radius"] = self.radius

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> MetricResult:
        method_names = list(group.keys())
        result = cls(method_names, radius=group.attrs["radius"])
        result.data = {m_name: np.array(group[m_name]) for m_name in method_names}
        return result
