from attrbench.suite import SuiteResult
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
from scripts.statistics.util import get_dfs, cohend, plot_wilcoxon_result, emp_power_curve, wilcoxon_tests
import os
from os import path
from tqdm import tqdm
import logging
from collections import defaultdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("-e", "--effect-size-measure", type=str, default="cohend", choices=["cohend", "meandiff"])
    args = parser.parse_args()

    # Constant parameters, might be moved to args if necessary
    IGNORE_METHODS = ["Random_pos_only", "GradCAM_no_relu", "GuidedGradCAM_no_relu"]
    BASELINE = "Random"
    PWR_ITERATIONS = 1000
    ALPHA = 0.01
    PWR_N_RANGE = np.arange(10, 256, 10)
    PWR_TOLERANCE = 0.8
    plt.rcParams["figure.dpi"] = 140
    res_obj = SuiteResult.load_hdf(args.file)

    if not path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S")

    # Extract necessary metric parameters
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
        }
    }

    for i, metric_group in enumerate(param_dict):
        logging.info(f"{metric_group}... ({i + 1}/{len(param_dict.keys())})")
        params = param_dict[metric_group]
        group_out_dir = path.join(args.out_dir, metric_group)
        if not path.isdir(group_out_dir):
            os.makedirs(group_out_dir)
        # variant -> (metric_name -> (method_name -> es/pv))
        effect_sizes, pvalues = defaultdict(dict), defaultdict(dict)

        prog = tqdm(params["metric_names"])
        for j, metric_name in enumerate(prog):
            dfs = get_dfs(res_obj, metric_name, BASELINE, IGNORE_METHODS)
            for variant, (df, baseline, inverted) in dfs.items():
                prog.set_postfix({"metric": metric_name, "variant": variant})
                variant_out_dir = path.join(group_out_dir, variant)
                if not path.isdir(variant_out_dir):
                    os.makedirs(variant_out_dir)

                # Compute effect sizes and p-values
                mes, mpv = wilcoxon_tests(df, baseline, args.effect_size_measure, inverted)
                effect_sizes[variant][metric_name] = mes
                pvalues[variant][metric_name] = mpv

                # Compute power curves for significant methods
                power_curves = {}
                for method_name in mpv.keys():
                    if mpv[method_name] < ALPHA:
                        effect_size = cohend(df[method_name].to_numpy(), baseline.to_numpy())
                        power_curves[method_name] = emp_power_curve(df[method_name].to_numpy(),
                                                                    baseline.to_numpy(),
                                                                    effect_size, PWR_ITERATIONS, PWR_N_RANGE, inverted,
                                                                    PWR_TOLERANCE,
                                                                    ALPHA)
                fig, ax = plt.subplots()
                for method_name, power_curve in power_curves.items():
                    ax.plot(PWR_N_RANGE, power_curve, label=method_name)
                fig.legend()
                fig.savefig(path.join(variant_out_dir, f"{metric_name.replace('.', '_')}_power.png"))

        for variant in effect_sizes.keys():
            es_df = pd.DataFrame.from_dict(effect_sizes[variant])
            pv_df = pd.DataFrame.from_dict(pvalues[variant])
            fig, axs = plot_wilcoxon_result(es_df, pv_df, params["labels"], ALPHA)
            fig.savefig(path.join(group_out_dir, variant, f"{metric_group}_summary.png"))
