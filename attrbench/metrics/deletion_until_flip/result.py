from attrbench.metrics import AbstractMetricResult
from attrbench.lib import NDArrayTree
import pandas as pd
import numpy as np
import h5py
from typing import List, Dict, Tuple


class DeletionUntilFlipResult(AbstractMetricResult):
    inverted = True

    def __init__(self, method_names: List[str], maskers: List[str]):
        super().__init__(method_names)
        self.maskers = maskers
        self.tree = NDArrayTree([
            ("masker", maskers),
            ("method", method_names)
        ])

    def append(self, data: Dict):
        self.tree.append(data)

    def add_to_hdf(self, group: h5py.Group):
        for masker in self.maskers:
            masker_group = group.create_group(masker)
            for method_name in self.method_names:
                masker_group.create_dataset(method_name, data=self.tree.get(
                    select=dict(masker=[masker], method=[method_name])))

    @classmethod
    def load_from_hdf(cls, group: h5py.Group):
        maskers = list(group.keys())
        method_names = list(group[maskers[0]].keys())
        result = cls(method_names, maskers)
        result.tree = NDArrayTree.load_from_hdf(["masker", "method"], group)
        return result

    def get_df(self, mode="raw", include_baseline=False, masker: str = "constant") -> Tuple[pd.DataFrame, bool]:
        raw_results = pd.DataFrame.from_dict(
            self.tree.get(
                postproc_fn=lambda x: np.squeeze(x, axis=-1),
                exclude=dict(method=["_BASELINE"]),
                select=dict(masker=[masker])
            )[masker]
        )
        baseline_results = pd.DataFrame(self.tree.get(
            postproc_fn=lambda x: np.squeeze(x, axis=-1),
            select=dict(method=["_BASELINE"], masker=[masker])
        )[masker]["_BASELINE"])
        if include_baseline:
            raw_results["Baseline"] = baseline_results.iloc[:, 0]
        if mode == "raw":
            return raw_results, self.inverted
        elif mode == "single_dist":
            return raw_results.sub(baseline_results.iloc[:, 0], axis=0), self.inverted
        else:
            baseline_avg = baseline_results.mean(axis=1)
            if mode == "raw_dist":
                return raw_results.sub(baseline_avg, axis=0), self.inverted
            elif mode == "std_dist":
                return raw_results \
                           .sub(baseline_avg, axis=0) \
                           .div(baseline_results.std(axis=1), axis=0).fillna(0), \
                       self.inverted
            else:
                raise ValueError(f"Invalid value for argument mode: {mode}. Must be raw, raw_dist or std_dist.")
