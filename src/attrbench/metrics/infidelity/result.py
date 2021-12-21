from typing import List
import pandas as pd
import numpy as np

import h5py

from attrbench.metrics import AbstractMetricResult
from attrbench.lib import NDArrayTree


class InfidelityResult(AbstractMetricResult):
    inverted = True

    def __init__(self, method_names: List[str], perturbation_generators: List[str],
                 activation_fns: List[str]):
        super().__init__(method_names)
        self._tree = NDArrayTree([
            ("perturbation_generator", perturbation_generators),
            ("activation_fn", activation_fns),
            ("method", method_names)
        ])

    @classmethod
    def load_from_hdf(cls, group: h5py.Group):
        pert_gens = list(group.keys())
        activation_fns = list(group[pert_gens[0]].keys())
        method_names = list(group[pert_gens[0]][activation_fns[0]].keys())
        result = cls(method_names, pert_gens, activation_fns)
        result._tree = NDArrayTree.load_from_hdf(["perturbation_generator", "activation_fn", "method"], group)
        return result

    def get_df(self, mode="raw", include_baseline=False, perturbation_generator="gaussian",
               activation_fn="linear"):
        def _squeeze(x):
            return np.squeeze(x, axis=-1)

        raw_results = pd.DataFrame.from_dict(
            self.tree.get(
                postproc_fn=_squeeze,
                exclude=dict(method=["_BASELINE"]),
                select=dict(perturbation_generator=[perturbation_generator],
                            activation_fn=[activation_fn])
            )[perturbation_generator][activation_fn]
        )
        baseline_results = pd.DataFrame(self.tree.get(
            postproc_fn=_squeeze,
            select=dict(perturbation_generator=[perturbation_generator],
                        activation_fn=[activation_fn],
                        method=["_BASELINE"])
        )[perturbation_generator][activation_fn]["_BASELINE"])
        res, _ = self._get_df(raw_results, baseline_results, mode, include_baseline)
        return res, self.inverted
