from __future__ import annotations
import h5py
from typing import List


class MetricResult:
    def __init__(self, method_names: List[str]):
        self.inverted = False
        self.method_names = method_names

    def add_to_hdf(self, group: h5py.Group):
        raise NotImplementedError

    def append(self, *args):
        raise NotImplementedError

    # TODO make sure there is a return statement everywhere
    # TODO remove code duplication in metric results
    @staticmethod
    def load_from_hdf(self, group: h5py.Group) -> MetricResult:
        raise NotImplementedError
