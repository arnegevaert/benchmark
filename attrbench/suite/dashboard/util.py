import numpy as np
from itertools import product
from attrbench.suite import SuiteResult

def get_dfs(res_obj: SuiteResult, mode: str = None, masker: str =None,activation:str = None, infid_log=False):
    res = {}

    for metric_name,dfs in res_obj.metric_results.items():
        if metric_name =='infidelity':
            res[metric_name] = dfs.get_df(mode=mode,log=infid_log) # ignore
        elif metric_name=='deletion_until_flip':
            res[metric_name]=dfs.get_df(mode=mode,masker=masker)
        else:
            res[metric_name]=dfs.get_df(mode=mode,masker=masker,activation_fn=activation)
    return res

def get_metric_df(metric_name, res_obj: SuiteResult, mode: str, infid_log=False):
    m_res = res_obj.metric_results[metric_name]
    if metric_name == "deletion_until_flip":

        df= {
            f"{masker}": m_res.get_df(mode=mode, masker=masker)
            for masker in ("constant", "random", "blur")
        }
    elif metric_name == "infidelity":
        df = {
            f"{pert_gen} - {afn} - {loss_fn}": m_res.get_df(mode=mode, perturbation_generator=pert_gen,
                                                                activation_fn=afn, loss_fn=loss_fn,
                                                                log=infid_log if loss_fn == "mse" else False)
            for (pert_gen, afn, loss_fn) in product(("gaussian", "square", "segment"),
                                                    ("linear", "softmax"),
                                                    ("mse", "normalized_mse", "corr"))
        }
    else:
        df = {
            f"{afn} - {masker}": m_res.get_df(mode=mode, activation_fn=afn, masker=masker)
            for afn, masker in product(("linear", "softmax"), ("constant", "random", "blur"))
        }

    return df

def _interval_metric(a, b):
    return (a - b) ** 2


def krippendorff_alpha(data):
    # Assumptions: no missing values, interval metric, data is numpy array ([observers, samples])
    # Assuming no missing values, each column is a unit, and the number of pairable values is m*n
    pairable_values = data.shape[0] * data.shape[1]

    # Calculate observed disagreement
    observed_disagreement = 0.
    for col in range(data.shape[1]):
        unit = data[:, col].reshape(1, -1)
        observed_disagreement += np.sum(_interval_metric(unit, unit.T))
    observed_disagreement /= (pairable_values * (data.shape[0] - 1))

    # Calculate expected disagreement
    expected_disagreement = 0.
    for col1 in range(data.shape[1]):
        unit1 = data[:, col1].reshape(1, -1)
        for col2 in range(data.shape[1]):
            unit2 = data[:, col2].reshape(1, -1)
            expected_disagreement += np.sum(_interval_metric(unit1, unit2.T))
    expected_disagreement /= (pairable_values * (pairable_values - 1))
    return 1. - (observed_disagreement / expected_disagreement)
