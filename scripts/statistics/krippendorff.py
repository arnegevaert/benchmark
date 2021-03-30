import argparse
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from attrbench.suite import SuiteResult
from os import path
import os
import logging
from scripts.statistics.util import get_dfs, krippendorff_alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("out_file", type=str)
    args = parser.parse_args()

    # Constant parameters, might be moved to args if necessary
    IGNORE_METHODS = ["Random_pos_only", "GradCAM_no_relu", "GuidedGradCAM_no_relu"]
    BASELINE = "Random"
    PWR_ITERATIONS = 1000
    ALPHA = 0.01
    plt.rcParams["figure.dpi"] = 140
    res_obj = SuiteResult.load_hdf(args.file)

    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S")

    param_dict = {
        "infidelity": {
            "metric_names": [f"infidelity_{pert}" for pert in ("gaussian", "seg", "sq")],
            "labels": ("Gaussian", "Segment", "Square"),
        },
        "deletion": {
            "metric_names": [f"masker_constant.deletion", "masker_blur.deletion", "masker_random.deletion"],
            "labels": ("Constant", "Blur", "Random"),
        },
        "insertion": {
            "metric_names": [f"masker_constant.insertion", "masker_blur.insertion", "masker_random.insertion"],
            "labels": ("Constant", "Blur", "Random"),
        },
        "irof": {
            "metric_names": [f"masker_constant.irof", "masker_blur.irof", "masker_random.irof"],
            "labels": ("Constant", "Blur", "Random"),
        },
        "iiof": {
            "metric_names": [f"masker_constant.iiof", "masker_blur.iiof", "masker_random.iiof"],
            "labels": ("Constant", "Blur", "Random"),
        },
        "sens-n": {
            "metric_names": [f"masker_constant.sensitivity_n", "masker_blur.sensitivity_n", "masker_random.sensitivity_n"],
            "labels": ("Constant", "Blur", "Random"),
        },
        "seg-sens-n": {
            "metric_names": [f"masker_constant.seg_sensitivity_n", "masker_blur.seg_sensitivity_n",
                             "masker_random.seg_sensitivity_n"],
            "labels": ("Constant", "Blur", "Random"),
        }
    }

    k_a = defaultdict(dict)
    for i, metric_group in enumerate(param_dict):
        params = param_dict[metric_group]
        for metric_name in params["metric_names"]:
            dfs = get_dfs(res_obj, metric_name, BASELINE, IGNORE_METHODS)
            for variant, (df, baseline, inverted) in dfs.items():
                activation = variant.split("_")[1] if "_" in variant else variant
                metric_variant_name = f"{metric_name}_{variant.split('_')[0]}" if '_' in variant else metric_name
                metric_variant_name = metric_variant_name.replace('.', '_')
                k_a[activation][metric_variant_name] = krippendorff_alpha(df.to_numpy())
    k_a = pd.DataFrame.from_dict(k_a)
    fig, ax = plt.subplots()
    k_a.plot.barh(figsize=(10, 10), ax=ax)
    fig.tight_layout()
    fig.savefig(args.out_file)
