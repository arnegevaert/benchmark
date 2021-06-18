from itertools import product
from attrbench.suite import SuiteResult


def get_default_dfs(res_obj: SuiteResult, mode: str, infid_log=False):
    res = {
        metric_name: res_obj.metric_results[metric_name].get_df(mode=mode)
        for metric_name in res_obj.metric_results.keys() if metric_name != "infidelity"
    }
    for infid_type in ("square", "noisy_bl", "gaussian"):
        res[f"infidelity-{infid_type}"] = res_obj.metric_results["infidelity"].get_df(mode=mode, log=infid_log,
                                                                                      perturbation_generator=infid_type)
    return res


def get_all_dfs(res_obj: SuiteResult, mode: str, infid_log=False):
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
                                                activation_fn=afn,
                                                log=infid_log)
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
