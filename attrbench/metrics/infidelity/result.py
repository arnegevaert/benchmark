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
        self.perturbation_generators = perturbation_generators
        self.activation_fns = activation_fns
        self.loss_fns = loss_fns
        self.tree = NDArrayTree([
            ("perturbation_generator", perturbation_generators),
            ("activation_fn", activation_fns),
            ("loss_fn", loss_fns),
            ("method", method_names + ["_BASELINE"])
        ])

    def append(self, data: Dict, **kwargs):
        self.tree.append(data, **kwargs)

    def add_to_hdf(self, group: h5py.Group):
        for pert_gen in self.perturbation_generators:
            pert_group = group.create_group(pert_gen)
            for loss in self.loss_fns:
                loss_group = pert_group.create_group(loss)
                for afn in self.activation_fns:
                    afn_group = loss_group.create_group(afn)
                    for m_name in self.method_names:
                        ds = afn_group.create_dataset(m_name, data=self.tree.get(
                            perturbation_generator=pert_gen,
                            loss_fn=loss,
                            activation_fn=afn,
                            method=m_name
                        ))
                        ds.attrs["inverted"] = self.inverted[loss]

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
