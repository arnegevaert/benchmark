from typing import List, Dict
import pandas as pd

import h5py
import numpy as np

from attrbench.metrics import AbstractMetricResult


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
        self.method_data = {
            pert_gen: {
                loss: {
                    afn: {
                        m_name: None for m_name in method_names
                    } for afn in activation_fns} for loss in loss_fns} for pert_gen in perturbation_generators}
        self.baseline_data = {
            pert_gen: {
                loss: {
                    afn: None for afn in activation_fns
                } for loss in loss_fns} for pert_gen in perturbation_generators}

    # TODO We need a nested dictionary datastructure that handles appending recursively
    def append(self, method_results: Dict, baseline_results: Dict):
        for pert_gen in self.perturbation_generators:
            for loss in self.loss_fns:
                for afn in self.activation_fns:
                    cur_data = self.baseline_data[pert_gen][loss][afn]
                    if cur_data is not None:
                        self.baseline_data[pert_gen][loss][afn] = np.concatenate(
                            [cur_data, baseline_results[pert_gen][loss][afn]])
                    else:
                        self.baseline_data[pert_gen][loss][afn] = baseline_results[pert_gen][loss][afn]

                    for method_name in self.method_names:
                        cur_data = self.method_data[pert_gen][loss][afn][method_name]
                        if cur_data is not None:
                            self.method_data[pert_gen][loss][afn][method_name] = np.concatenate([
                                cur_data, method_results[pert_gen][method_name][loss][afn]
                            ])
                        else:
                            self.method_data[pert_gen][loss][afn][method_name] = \
                            method_results[pert_gen][method_name][loss][afn]

    def add_to_hdf(self, group: h5py.Group):
        for pert_gen in self.perturbation_generators:
            pert_group = group.create_group(pert_gen)
            for loss in self.loss_fns:
                loss_group = pert_group.create_group(loss)
                for afn in self.activation_fns:
                    afn_group = loss_group.create_group(afn)
                    for m_name in self.method_names:
                        ds = afn_group.create_dataset(m_name, data=self.method_data[pert_gen][loss][afn][m_name])
                        ds.attrs["inverted"] = self.inverted[loss]
                    ds = afn_group.create_dataset("_BASELINE", data=self.baseline_data[pert_gen][loss][afn])
                    ds.attrs["inverted"] = self.inverted[loss]

    @classmethod
    def load_from_hdf(cls, group: h5py.Group):
        pert_gens = list(group.keys())
        loss_fns = list(group[pert_gens[0]].keys())
        activation_fns = list(group[pert_gens[0]][loss_fns[0]].keys())
        method_names = list(group[pert_gens[0]][loss_fns[0]][activation_fns[0]].keys())
        result = cls(method_names, pert_gens, activation_fns, loss_fns)
        result.method_data = {
            pert_gen: {
                loss: {
                    afn: {
                        m_name: None for m_name in method_names
                    } for afn in activation_fns} for loss in loss_fns} for pert_gen in pert_gens}
        return result

    def get_df(self, *, pert_gen=None, loss=None, activation=None):
        data = {m_name: self.method_data[pert_gen][loss][activation][m_name].squeeze()
                for m_name in self.method_names}
        df = pd.DataFrame.from_dict(data)
        return df, self.inverted
