from attrbench.suite import SuiteResult
import numpy as np


def get_default_dfs(res_obj: SuiteResult, mode: str, activation_fn="linear", masker="constant"):
    # Add simple metrics
    res = {
        metric_name: res_obj.metric_results[metric_name].get_df(mode=mode, activation_fn=activation_fn, masker=masker)
        for metric_name in ["impact_coverage", "minimal_subset_deletion", "minimal_subset_insertion",
                            "sensitivity_n", "seg_sensitivity_n", "max_sensitivity"]
        if metric_name in res_obj.metric_results.keys()
    }

    # Add deletion/insertion
    limit = 50
    res["deletion_morf"] = res_obj.metric_results["deletion_morf"].get_df(mode=mode, columns=np.arange(limit),
                                                                          activation_fn=activation_fn, masker=masker)
    res["deletion_lerf"] = res_obj.metric_results["deletion_lerf"].get_df(mode=mode, columns=np.arange(limit),
                                                                          activation_fn=activation_fn, masker=masker)
    ins_morf = res_obj.metric_results["deletion_lerf"].get_df(mode=mode, columns=np.arange(100-limit, 100),
                                                              activation_fn=activation_fn, masker=masker)
    res["insertion_morf"] = (ins_morf[0][::-1], ins_morf[1])
    ins_lerf = res_obj.metric_results["deletion_morf"].get_df(mode=mode, columns=np.arange(100-limit, 100),
                                                              activation_fn=activation_fn, masker=masker)
    res["insertion_lerf"] = (ins_lerf[0][::-1], ins_lerf[1])

    # Add IROF/IIOF
    limit = 50
    res["irof_morf"] = res_obj.metric_results["irof_morf"].get_df(mode=mode, columns=np.arange(limit),
                                                                  activation_fn=activation_fn, masker=masker)
    res["irof_lerf"] = res_obj.metric_results["irof_lerf"].get_df(mode=mode, columns=np.arange(limit),
                                                                  activation_fn=activation_fn, masker=masker)
    iiof_morf = res_obj.metric_results["irof_lerf"].get_df(mode=mode, columns=np.arange(100-limit, 100),
                                                           activation_fn=activation_fn, masker=masker)
    res["iiof_morf"] = (iiof_morf[0][::-1], iiof_morf[1])
    iiof_lerf = res_obj.metric_results["irof_morf"].get_df(mode=mode, columns=np.arange(100-limit, 100),
                                                           activation_fn=activation_fn, masker=masker)
    res["iiof_lerf"] = (iiof_lerf[0][::-1], iiof_lerf[1])

    for infid_type in ("square", "noisy_bl"):
        res[f"infidelity_{infid_type}"] = res_obj.metric_results["infidelity"].get_df(mode=mode,
                                                                                      perturbation_generator=infid_type,
                                                                                      activation_fn=activation_fn)
    return res
