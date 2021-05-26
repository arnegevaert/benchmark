from itertools import product
from attrbench.suite import SuiteResult


def get_default_dfs(res_obj: SuiteResult, mode: str, infid_log=False):
    res = {
        metric_name: res_obj.metric_results[metric_name].get_df(mode=mode)
        for metric_name in res_obj.metric_results.keys() if metric_name != "infidelity"
    }
    res["infidelity"] = res_obj.metric_results["infidelity"].get_df(mode=mode, log=infid_log)
    return res


def get_metric_dfs(res_obj: SuiteResult, mode: str, infid_log=False):
    dfs = {}
    for m_name in ("deletion", "insertion", "irof", "iiof", "sensitivity_n", "seg_sensitivity_n"):
        m_res = res_obj.metric_results[m_name]
        dfs[m_name] = {
            f"{afn} - {masker}": m_res.get_df(mode=mode, activation_fn=afn, masker=masker)
            for afn, masker in product(("linear", "softmax"), ("constant", "random", "blur"))
        }

    del_flip_res = res_obj.metric_results["minimal_subset"]
    dfs["minimal_subset"] = {
        f"{masker}": del_flip_res.get_df(mode=mode, masker=masker)
        for masker in ("constant", "random", "blur")
    }

    infid_res = res_obj.metric_results["infidelity"]
    dfs["infidelity"] = {
        f"{pert_gen} - {afn} - {loss_fn}": infid_res.get_df(mode=mode, perturbation_generator=pert_gen,
                                                            activation_fn=afn, loss_fn=loss_fn, log=infid_log if loss_fn == "mse" else False)
        for (pert_gen, afn, loss_fn) in product(("gaussian", "square", "segment"),
                                                ("linear", "softmax"),
                                                ("mse", "normalized_mse", "corr"))
    }
    return dfs

