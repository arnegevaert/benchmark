from attrbench.suite import SuiteResult
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from scipy import stats
import numpy as np
from scripts.statistics.util import get_dfs, cohend, plot_wilcoxon_result, emp_power_curve


def wilcoxon_tests(df, baseline, effect_size_measure):
    pvalues, effect_sizes = {}, {}
    for method_name in df:
        method_results = df[method_name].to_numpy()
        statistic, pvalue = stats.wilcoxon(method_results, baseline,
                                           alternative="less" if inverted else "greater")
        pvalues[method_name] = pvalue
        if effect_size_measure == "cohend":
            effect_sizes[method_name] = cohend(method_results, baseline)
        elif effect_size_measure == "meandiff":
            effect_sizes[method_name] = np.mean(method_results) - np.mean(baseline)
        else:
            raise ValueError("Variant must be mse or corr")
    return effect_sizes, pvalues


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("metric", type=str)
    parser.add_argument("-m", "--mode", type=str)
    parser.add_argument("-a", "--activation", type=str)
    parser.add_argument("-e", "--effect-size-measure", type=str, default="cohend", choices=["cohend", "meandiff"])
    args = parser.parse_args()

    # Constant parameters, might be moved to args if necessary
    IGNORE_METHODS = ["Random_pos_only", "GradCAM_no_relu", "GuidedGradCAM_no_relu"]
    BASELINE = "Random"
    PILOT_ROWS = 128
    PWR_ITERATIONS = 100
    PWR_ALPHA = 0.01
    PWR_N_RANGE = np.arange(5, 128, 5)
    PWR_TOLERANCE = 0.9
    plt.rcParams["figure.dpi"] = 140
    res_obj = SuiteResult.load_hdf(args.file)

    # Extract necessary metric parameters
    param_dict = {
        "infidelity": {
            "metric_names": [f"infidelity_{pert}" for pert in ("gaussian", "seg", "sq")],
            "labels": ("Gaussian", "Segment", "Square"),
            "title": f"Infidelity - {args.mode} - {args.activation}"
        },
        "deletion": {
            "metric_names": [f"masker_constant.deletion", "masker_blur.deletion", "masker_random.deletion"],
            "labels": ("Constant", "Blur", "Random"),
            "title": f"Deletion - {args.mode} - {args.activation}"
        },
        "insertion": {
            "metric_names": [f"masker_constant.insertion", "masker_blur.insertion", "masker_random.insertion"],
            "labels": ("Constant", "Blur", "Random"),
            "title": f"Insertion - {args.mode} - {args.activation}"
        },
        "irof": {
            "metric_names": [f"masker_constant.irof", "masker_blur.irof", "masker_random.irof"],
            "labels": ("Constant", "Blur", "Random"),
            "title": f"IROF - {args.mode} - {args.activation}"
        },
        "iiof": {
            "metric_names": [f"masker_constant.iiof", "masker_blur.iiof", "masker_random.iiof"],
            "labels": ("Constant", "Blur", "Random"),
            "title": f"IIOF - {args.mode} - {args.activation}"
        }
    }[args.metric]

    effect_sizes, pvalues = {}, {}
    for i, metric_name in enumerate(param_dict["metric_names"]):
        # Compute effect sizes and p-values
        df, baseline, inverted = get_dfs(res_obj, metric_name, BASELINE, args.mode,
                                         args.activation, IGNORE_METHODS)
        mes, mpv = wilcoxon_tests(df.iloc[:PILOT_ROWS, :], baseline[:PILOT_ROWS], args.effect_size_measure)
        effect_sizes[metric_name] = mes
        pvalues[metric_name] = mpv

        # Compute power curves for significant methods
        power_curves = {}
        for method_name in mpv.keys():
            if mpv[method_name] < PWR_ALPHA:
                pilot_effect_size = cohend(df[method_name].iloc[:PILOT_ROWS], baseline.iloc[:PILOT_ROWS])
                power_curves[method_name] = emp_power_curve(df[method_name].iloc[PILOT_ROWS:].to_numpy(),
                                                            baseline.iloc[PILOT_ROWS:].to_numpy(),
                                                            pilot_effect_size, PWR_ITERATIONS, PWR_N_RANGE, inverted,
                                                            PWR_TOLERANCE,
                                                            PWR_ALPHA)
        for method_name, power_curve in power_curves.items():
            plt.plot(PWR_N_RANGE, power_curve, label=method_name)
        plt.legend()
        plt.title(f"{param_dict['title']} - {param_dict['labels'][i]} - Power analysis")
        plt.show()

    effect_sizes = pd.DataFrame.from_dict(effect_sizes)
    pvalues = pd.DataFrame.from_dict(pvalues)
    plot_wilcoxon_result(param_dict["title"], effect_sizes, pvalues, param_dict["labels"])
