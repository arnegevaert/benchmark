from itertools import product
from attrbench.suite import SuiteResult
import numpy as np


def get_default_dfs(res_obj: SuiteResult, mode: str):
    # Add simple metrics
    res = {
        metric_name: res_obj.metric_results[metric_name].get_df(mode=mode)
        for metric_name in ["impact_coverage", "minimal_subset_deletion", "minimal_subset_insertion",
                            "sensitivity_n", "seg_sensitivity_n"] if metric_name in res_obj.metric_results.keys()
    }

    # Add deletion/insertion
    limit = 50
    res["deletion_morf"] = res_obj.metric_results["deletion_morf"].get_df(mode=mode, columns=np.arange(limit))
    res["deletion_lerf"] = res_obj.metric_results["deletion_lerf"].get_df(mode=mode, columns=np.arange(limit))
    ins_morf = res_obj.metric_results["deletion_lerf"].get_df(mode=mode, columns=np.arange(100-limit, 100))
    res["insertion_morf"] = (ins_morf[0][::-1], ins_morf[1])
    ins_lerf = res_obj.metric_results["deletion_morf"].get_df(mode=mode, columns=np.arange(100-limit, 100))
    res["insertion_lerf"] = (ins_lerf[0][::-1], ins_lerf[1])

    # Add IROF/IIOF
    limit = 50
    res["irof_morf"] = res_obj.metric_results["irof_morf"].get_df(mode=mode, columns=np.arange(limit))
    res["irof_lerf"] = res_obj.metric_results["irof_lerf"].get_df(mode=mode, columns=np.arange(limit))
    iiof_morf = res_obj.metric_results["irof_lerf"].get_df(mode=mode, columns=np.arange(100-limit, 100))
    res["iiof_morf"] = (iiof_morf[0][::-1], iiof_morf[1])
    iiof_lerf = res_obj.metric_results["irof_morf"].get_df(mode=mode, columns=np.arange(100-limit, 100))
    res["iiof_lerf"] = (iiof_lerf[0][::-1], iiof_lerf[1])

    for infid_type in ("square", "noisy_bl", "gaussian"):
        res[f"infidelity_{infid_type}"] = res_obj.metric_results["infidelity"].get_df(mode=mode,
                                                                                      perturbation_generator=infid_type)
    return res


def get_all_dfs(res_obj: SuiteResult, mode: str):
    dfs = {}
    for m_name in ("deletion_morf", "irof_morf", "deletion_lerf", "irof_lerf", "sensitivity_n", "seg_sensitivity_n"):
        m_res = res_obj.metric_results[m_name]
        dfs[m_name] = {
            f"{afn} - {masker}": m_res.get_df(mode=mode, activation_fn=afn, masker=masker)
            for afn, masker in product(("linear", "softmax"), ("constant", "random", "blur"))
        }

    # Minimal subset insertion/deletion
    for m_name in ("minimal_subset_insertion", "minimal_subset_deletion"):
        del_flip_res = res_obj.metric_results[m_name]
        dfs[m_name] = {
            f"{masker}": del_flip_res.get_df(mode=mode, masker=masker)
            for masker in ("constant", "random", "blur")
        }

    # Infidelity
    infid_res = res_obj.metric_results["infidelity"]
    dfs["infidelity"] = {
        f"{pert_gen} - {afn}": infid_res.get_df(mode=mode, perturbation_generator=pert_gen,
                                                activation_fn=afn)
        for (pert_gen, afn) in product(("gaussian", "square", "noisy_bl"),
                                       ("linear", "softmax"))
    }

    # Impact Coverage (if present)
    if "impact_coverage" in res_obj.metric_results.keys():
        ic_res = res_obj.metric_results["impact_coverage"]
        dfs["impact_coverage"] = {
            "Impact Coverage": ic_res.get_df(mode=mode)
        }
    return dfs
