from typing import List

import h5py
import numpy as np

from attrbench.metrics import BasicMetricResult
from attrbench.lib import NDArrayTree


class MaxSensitivityResult(BasicMetricResult):
    inverted = True

    def __init__(self, method_names: List[str], radius: float):
        super().__init__(method_names)
        self.inverted = True
        self.radius = radius

    def add_to_hdf(self, group: h5py.Group):
        super().add_to_hdf(group)
        group.attrs["radius"] = self.radius

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> BasicMetricResult:
        method_names = list(group.keys())
        result = cls(method_names, radius=group.attrs["radius"])
        result._tree = NDArrayTree.load_from_hdf(["method"], group)
        return result
