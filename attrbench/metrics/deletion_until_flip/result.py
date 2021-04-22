from attrbench.metrics import BasicMetricResult
import pandas as pd
import numpy as np
import h5py
from typing import List, Dict, Tuple


class DeletionUntilFlipResult(BasicMetricResult):
    inverted = True

    def __init__(self, method_names: List[str], maskers: List[str]):
        super().__init__(method_names)
        self.maskers = maskers
        self.data = {
            masker: {
                m_name: None for m_name in method_names
            } for masker in maskers}

    def append(self, method_name, batch: Dict):
        for masker in batch.keys():
            cur_data = self.data[masker][method_name]
            if cur_data is not None:
                self.data[masker][method_name] = np.concatenate([cur_data, batch[masker]], axis=0)
            else:
                self.data[masker][method_name] = batch[masker]

    def add_to_hdf(self, group: h5py.Group):
        for masker in self.maskers:
            masker_group = group.create_group(masker)
            for method_name in self.method_names:
                ds = masker_group.create_dataset(method_name, data=self.data[masker][method_name])
                ds.attrs["inverted"] = self.inverted

    @classmethod
    def load_from_hdf(cls, group: h5py.Group):
        maskers = list(group.keys())
        method_names = list(group[maskers[0]].keys())
        result = cls(method_names, maskers)
        result.data = {
            masker: {
                m_name: np.array(group[masker][m_name]) for m_name in method_names
            } for masker in maskers}
        return result

    def get_df(self, *, masker=None) -> Tuple[pd.DataFrame, bool]:
        data = {m_name: self._aggregate(self.data[masker][m_name].squeeze())
                for m_name in self.method_names}
        df = pd.DataFrame.from_dict(data)
        return df, self.inverted
