from typing import List, Dict
import pandas as pd

import h5py
import numpy as np

from attrbench.metrics import MetricResult


class InfidelityResult(MetricResult):
    inverted = {
        "mse": True,
        "normalized_mse": True,
        "corr": False
    }

    def __init__(self, method_names: List[str], perturbation_modes: List[str],
                 activation_fns: List[str], loss_fns: List[str]):
        super().__init__(method_names)
        self.perturbation_modes = perturbation_modes
        self.activation_fns = activation_fns
        self.loss_fns = loss_fns
        self.data = {
            pert_mode: {
                loss: {
                    afn: {
                        m_name: None for m_name in method_names
                    } for afn in activation_fns} for loss in loss_fns} for pert_mode in perturbation_modes}

    def append(self, method_name, batch: Dict):
        for pert_mode in batch.keys():
            for loss in batch[pert_mode].keys():
                for afn in batch[pert_mode][loss].keys():
                    cur_data = self.data[pert_mode][loss][afn][method_name]
                    if cur_data is not None:
                        self.data[pert_mode][loss][afn][method_name] = np.concatenate(
                            [cur_data, batch[pert_mode][loss][afn]], axis=0)
                    else:
                        self.data[pert_mode][loss][afn][method_name] = batch[pert_mode][loss][afn]

    def add_to_hdf(self, group: h5py.Group):
        for pert_mode in self.perturbation_modes:
            pert_group = group.create_group(pert_mode)
            for loss in self.loss_fns:
                loss_group = pert_group.create_group(loss)
                for afn in self.activation_fns:
                    afn_group = loss_group.create_group(afn)
                    for m_name in self.method_names:
                        ds = afn_group.create_dataset(m_name, data=self.data[pert_mode][loss][afn][m_name])
                        ds.attrs["inverted"] = self.inverted[loss]

    @classmethod
    def load_from_hdf(cls, group: h5py.Group):
        pert_modes = list(group.keys())
        loss_fns = list(group[pert_modes[0]].keys())
        activation_fns = list(group[pert_modes[0]][loss_fns[0]].keys())
        method_names = list(group[pert_modes[0]][loss_fns[0]][activation_fns[0]].keys())
        result = cls(method_names, pert_modes, activation_fns, loss_fns)
        result.data = {
            pert_mode: {
                loss: {
                    afn: {
                        m_name: None for m_name in method_names
                    } for afn in activation_fns} for loss in loss_fns} for pert_mode in pert_modes}
        return result

    def get_df(self, *, pert_mode=None, loss=None, activation=None):
        data = {m_name: self._aggregate(self.data[pert_mode][loss][activation][m_name].squeeze())
                for m_name in self.method_names}
        df = pd.DataFrame.from_dict(data)
        return df, self.inverted
