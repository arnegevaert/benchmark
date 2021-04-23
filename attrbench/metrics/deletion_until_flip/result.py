from attrbench.metrics import AbstractMetricResult
import pandas as pd
import numpy as np
import h5py
from typing import List, Dict, Tuple


class DeletionUntilFlipResult(AbstractMetricResult):
    inverted = True

    def __init__(self, method_names: List[str], maskers: List[str]):
        super().__init__(method_names)
        self.maskers = maskers
        self.method_data = {
            masker: {
                m_name: None for m_name in method_names
            } for masker in maskers}
        self.baseline_data = {masker: None for masker in maskers}

    def append(self, method_results: Dict, baseline_results: Dict):
        for masker in method_results.keys():
            # Append method results
            for method_name in method_results[masker].keys():
                cur_data = self.method_data[masker][method_name]
                if cur_data is not None:
                    self.method_data[masker][method_name] = np.concatenate(
                        [cur_data, method_results[masker][method_name]], axis=0)
                else:
                    self.method_data[masker][method_name] = method_results[masker][method_name]

            # Append baseline results
            cur_baseline_data = self.baseline_data[masker]
            if cur_baseline_data is not None:
                self.baseline_data[masker] = np.concatenate([cur_baseline_data, baseline_results[masker]], axis=0)
            else:
                self.baseline_data[masker] = baseline_results[masker]

    def add_to_hdf(self, group: h5py.Group):
        for masker in self.maskers:
            masker_group = group.create_group(masker)
            for method_name in self.method_names:
                ds = masker_group.create_dataset(method_name, data=self.method_data[masker][method_name])
                ds.attrs["inverted"] = self.inverted
            ds = masker_group.create_dataset("_BASELINE", data=self.baseline_data[masker])
            ds.attrs["inverted"] = self.inverted

    @classmethod
    def load_from_hdf(cls, group: h5py.Group):
        maskers = list(group.keys())
        method_names = list(group[maskers[0]].keys())
        result = cls(method_names, maskers)
        result.method_data = {
            masker: {
                m_name: np.array(group[masker][m_name]) for m_name in method_names
            } for masker in maskers}
        result.baseline_data = {masker: np.array(group[masker]["_BASELINE"]) for masker in maskers}
        return result

    def get_df(self, *, masker=None) -> Tuple[pd.DataFrame, bool]:
        data = {m_name: self._aggregate(self.method_data[masker][m_name].squeeze())
                for m_name in self.method_names}
        df = pd.DataFrame.from_dict(data)
        return df, self.inverted
