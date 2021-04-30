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
        self.tree = NDArrayTree([
            ("perturbation_generator", perturbation_generators),
            ("activation_fn", activation_fns),
            ("loss_fn", loss_fns),
            ("method", method_names)
        ])

    def append(self, data: Dict, **kwargs):
        self.tree.append(data, **kwargs)

    def add_to_hdf(self, group: h5py.Group):
        self.tree.add_to_hdf(group)

    @classmethod
    def load_from_hdf(cls, group: h5py.Group):
        pert_gens = list(group.keys())
        activation_fns = list(group[pert_gens[0]].keys())
        loss_fns = list(group[pert_gens[0]][activation_fns[0]].keys())
        method_names = list(group[pert_gens[0]][activation_fns[0]][loss_fns[0]].keys())
        result = cls(method_names, pert_gens, activation_fns, loss_fns)
        result.tree = NDArrayTree.load_from_hdf(["perturbation_generator", "activation_fn", "loss_fn", "method"], group)
        return result

    def get_df(self, loss_fn, **kwargs):
        return pd.DataFrame.from_dict(self.tree.get(postproc_fn=lambda x: np.squeeze(x, axis=-1), loss_fn=loss_fn, **kwargs)), self.inverted[loss_fn]
