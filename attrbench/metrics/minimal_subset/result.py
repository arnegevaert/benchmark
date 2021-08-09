from attrbench.metrics import AbstractMetricResult
from attrbench.lib import NDArrayTree
import pandas as pd
import numpy as np
import h5py
from typing import List, Tuple


class MinimalSubsetResult(AbstractMetricResult):
    inverted = True

    def __init__(self, method_names: List[str], maskers: List[str]):
        super().__init__(method_names)
        self.maskers = maskers
        self._tree = NDArrayTree([
            ("masker", maskers),
            ("method", method_names)
        ])

    @classmethod
    def load_from_hdf(cls, group: h5py.Group):
        maskers = list(group.keys())
        method_names = list(group[maskers[0]].keys())
        result = cls(method_names, maskers)
        result._tree = NDArrayTree.load_from_hdf(["masker", "method"], group)
        return result

    def get_df(self, mode="raw", include_baseline=False, masker: str = "constant", **kwargs) -> Tuple[pd.DataFrame, bool]:
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
        return self._get_df(raw_results, baseline_results, mode, include_baseline)
