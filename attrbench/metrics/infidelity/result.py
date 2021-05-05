from typing import List, Dict
import pandas as pd
import numpy as np

import h5py

from attrbench.metrics import AbstractMetricResult
from attrbench.lib import NDArrayTree


class InfidelityResult(AbstractMetricResult):
    inverted = {
        "mse": True,
        "normalized_mse": True,
        "corr": False
    }

    def __init__(self, method_names: List[str], perturbation_generators: List[str],
                 activation_fns: List[str], loss_fns: List[str]):
        super().__init__(method_names)
        self._tree = NDArrayTree([
            ("perturbation_generator", perturbation_generators),
            ("activation_fn", activation_fns),
            ("loss_fn", loss_fns),
            ("method", method_names)
        ])

    @classmethod
    def load_from_hdf(cls, group: h5py.Group):
        pert_gens = list(group.keys())
        activation_fns = list(group[pert_gens[0]].keys())
        loss_fns = list(group[pert_gens[0]][activation_fns[0]].keys())
        method_names = list(group[pert_gens[0]][activation_fns[0]][loss_fns[0]].keys())
        result = cls(method_names, pert_gens, activation_fns, loss_fns)
        result._tree = NDArrayTree.load_from_hdf(["perturbation_generator", "activation_fn", "loss_fn", "method"], group)
        return result

    def get_df(self, mode="raw", include_baseline=False, perturbation_generator="gaussian", activation_fn="linear", loss_fn="mse"):
        raw_results = pd.DataFrame.from_dict(
            self.tree.get(
                postproc_fn=lambda x: np.squeeze(x, axis=-1),
                exclude=dict(method=["_BASELINE"]),
                select=dict(perturbation_generator=[perturbation_generator],
                            activation_fn=[activation_fn],
                            loss_fn=[loss_fn])
            )[perturbation_generator][activation_fn][loss_fn]
        )
        baseline_results = pd.DataFrame(self.tree.get(
            postproc_fn=lambda x: np.squeeze(x, axis=-1),
            select=dict(perturbation_generator=[perturbation_generator],
                        activation_fn=[activation_fn],
                        loss_fn=[loss_fn],
                        method=["_BASELINE"])
        )[perturbation_generator][activation_fn][loss_fn]["_BASELINE"])
        return self._get_df(raw_results, baseline_results, mode, include_baseline)
