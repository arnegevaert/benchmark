from attrbench.metrics import AbstractMetricResult
from attrbench.lib import NDArrayTree
import pandas as pd
import h5py
from typing import List, Dict, Tuple


class DeletionUntilFlipResult(AbstractMetricResult):
    inverted = True

    def __init__(self, method_names: List[str], maskers: List[str]):
        super().__init__(method_names)
        self.maskers = maskers
        self.tree = NDArrayTree([
            ("masker", maskers),
            ("method", method_names + ["_BASELINE"])
        ])

    def append(self, data: Dict):
        self.tree.append(data)

    def add_to_hdf(self, group: h5py.Group):
        for masker in self.maskers:
            masker_group = group.create_group(masker)
            for method_name in self.method_names:
                ds = masker_group.create_dataset(method_name, data=self.tree.get(masker=masker, method=method_name))
                ds.attrs["inverted"] = self.inverted

    @classmethod
    def load_from_hdf(cls, group: h5py.Group):
        maskers = list(group.keys())
        method_names = list(group[maskers[0]].keys())
        result = cls(method_names, maskers)
        result.append(dict(group))
        return result

    def get_df(self, **kwargs) -> Tuple[pd.DataFrame, bool]:
        return pd.DataFrame.from_dict(self.tree.get(**kwargs)), self.inverted
