from typing import List, Dict
import pandas as pd

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
        loss_fns = list(group[pert_gens[0]].keys())
        activation_fns = list(group[pert_gens[0]][loss_fns[0]].keys())
        method_names = list(group[pert_gens[0]][loss_fns[0]][activation_fns[0]].keys())
        result = cls(method_names, pert_gens, activation_fns, loss_fns)
        result.append(dict(group))
        return result

    def get_df(self, **kwargs):
        return pd.DataFrame.from_dict(self.tree.get(**kwargs)), self.inverted
