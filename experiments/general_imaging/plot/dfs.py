from attrbench.suite import SuiteResult
import numpy as np


# Derive Deletion/Insertion from single Deletion run with full pixel range
def _derive_del_ins(res_obj: SuiteResult, mode: str, activation_fn="linear", masker="constant", limit=50):
    res = {}
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


def get_default_dfs(res_obj: SuiteResult, mode: str, activation_fn="linear", masker="constant"):
    # Add simple metrics
    res = {
        metric_name: res_obj.metric_results[metric_name].get_df(mode=mode, activation_fn=activation_fn, masker=masker)
        for metric_name in ["impact_coverage", "minimal_subset_deletion", "minimal_subset_insertion",
                            "sensitivity_n", "seg_sensitivity_n", "max_sensitivity", "deletion_morf", "deletion_lerf",
                            "insertion_morf", "insertion_lerf", "irof_morf", "irof_lerf"]
        if metric_name in res_obj.metric_results.keys()
    }

    for infid_type in ("square", "noisy_bl"):
        res[f"infidelity_{infid_type}"] = res_obj.metric_results["infidelity"].get_df(mode=mode,
                                                                                      perturbation_generator=infid_type,
                                                                                      activation_fn=activation_fn)
    return res


def get_all_dfs(res_obj: SuiteResult, mode: str):
    res = dict()
    activation_fns = ["linear"]
    if "impact_coverage" in res_obj.metric_results.keys():
        res["impact_coverage"] = res_obj.metric_results["impact_coverage"].get_df(mode=mode)
    res["max_sensitivity"] = res_obj.metric_results["max_sensitivity"].get_df(mode=mode)
    for metric_name in ["deletion_morf", "deletion_lerf", "insertion_morf", "insertion_lerf", "irof_morf", "irof_lerf",
                        "sensitivity_n", "seg_sensitivity_n"]:
        for masker in ["blur", "constant", "random"]:
            for activation in activation_fns:
                if metric_name not in ["sensitivity_n", "seg_sensitivity_n"]:
                    res[f"{metric_name}_{masker}_{activation}_norm"] = res_obj.metric_results[metric_name].get_df(
                        masker=masker, activation_fn=activation, mode=mode, normalize=True)
                    #res[f"{metric_name}_{masker}_{activation}"] = res_obj.metric_results[metric_name].get_df(
                    #    masker=masker, activation_fn=activation, mode=mode, normalize=False)
                else:
                    res[f"{metric_name}_{masker}_{activation}"] = res_obj.metric_results[metric_name].get_df(
                        masker=masker, activation_fn=activation, mode=mode)

    for infid_type in ["square", "noisy_bl"]:
        for activation in activation_fns:
            res[f"infidelity_{infid_type}_{activation}"] = res_obj.metric_results["infidelity"].get_df(
                perturbation_generator=infid_type, activation_fn=activation, mode=mode
            )

    for metric_name in ["minimal_subset_insertion", "minimal_subset_deletion"]:
        for masker in ["blur", "constant", "random"]:
            res[f"{metric_name}_{masker}"] = res_obj.metric_results[metric_name].get_df(masker=masker, mode=mode)
    return res
