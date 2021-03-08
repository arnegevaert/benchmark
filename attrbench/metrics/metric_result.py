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

    @staticmethod
    def load_from_hdf(self, group: h5py.Group):
        raise NotImplementedError
