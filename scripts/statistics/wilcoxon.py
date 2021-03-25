from attrbench.suite import SuiteResult
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


def cohend(x, y, type="mean"):
    pooled_std = np.sqrt(((x.shape[0] - 1) * np.var(x) + (y.shape[0] - 1) * np.var(y)) / (x.shape[0] + y.shape[0] - 2))
    if type == "mean":
        return (np.mean(x) - np.mean(y)) / pooled_std
    elif type == "median":
        return (np.median(x) - np.median(y)) / pooled_std
    else:
        raise ValueError("Type must be mean or median")


def plot_wilcoxon_result(title, effect_sizes, pvalues, labels):
    fig, axs = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [4, 1]})

    effect_sizes.plot.barh(figsize=(14, 6), title=f"{title} effect sizes", ax=axs[0])
    axs[0].legend(labels, loc='upper center', bbox_to_anchor=(0.5, -.05), ncol=3, fancybox=True,
                  shadow=True)
    axs[1].pcolor(pvalues < 0.05, cmap="RdYlGn", edgecolor="black")
    axs[1].set_title("p < 0.05")
    axs[1].set_yticks([])
    axs[1].set_xticks(np.arange(3) + 0.5)
    axs[1].tick_params(axis="x", rotation=45)
    axs[1].set_xticklabels(labels, ha="right")
    plt.show()


def wilcoxon_tests(res_obj, title, names, mode, activation, effect_size_measure, labels):
    # Infidelity
    all_pvalues, all_effect_sizes = {}, {}
    for name in names:
        pvalues, effect_sizes = {}, {}
        # Load dataframes from object
        df_dict = res_obj.metric_results[name].to_df()
        # Select the correct variant of the metric
        df = df_dict[f"{mode}_{activation}"]
        inverted = res_obj.metric_results[name].inverted[mode]
        # Take the first n rows for pilot study
        df = df.iloc[:PILOT_ROWS, :]
        # Remove the methods that we want to ignore
        df = df[df.columns.difference(IGNORE_METHODS)]
        baseline_results = df[BASELINE].to_numpy()
        for method_name in df.columns.difference([BASELINE]):
            method_results = df[method_name].to_numpy()
            statistic, pvalue = stats.wilcoxon(method_results, baseline_results,
                                               alternative="less" if inverted else "greater")
            pvalues[method_name] = pvalue
            # Probability of superiority effect size
            if effect_size_measure == "cohend":
                effect_sizes[method_name] = cohend(method_results, baseline_results)
            elif effect_size_measure == "meandiff":
                effect_sizes[method_name] = np.mean(method_results) - np.mean(baseline_results)
            else:
                raise ValueError("Variant must be mse or corr")
        all_pvalues[name] = pvalues
        all_effect_sizes[name] = effect_sizes
    all_pvalues = pd.DataFrame.from_dict(all_pvalues)
    all_effect_sizes = pd.DataFrame.from_dict(all_effect_sizes)
    plot_wilcoxon_result(title, all_effect_sizes, all_pvalues, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("metric", type=str)
    parser.add_argument("-m", "--mode", type=str, default="mse")
    parser.add_argument("-a", "--activation", type=str, default="linear")
    parser.add_argument("-e", "--effect-size-measure", type=str, default="cohend", choices=["cohend", "meandiff"])
    args = parser.parse_args()

    IGNORE_METHODS = ["Random_pos_only", "GradCAM_no_relu", "GuidedGradCAM_no_relu"]
    BASELINE = "Random"
    PILOT_ROWS = 128
    plt.rcParams["figure.dpi"] = 140
    res_obj = SuiteResult.load_hdf(args.file)

    if args.metric == "infidelity":
        infidelity_names = [f"infidelity_{pert}" for pert in ("gaussian", "seg", "sq")]
        wilcoxon_tests(res_obj, "Infidelity", infidelity_names, args.mode, args.activation, args.effect_size_measure, ("Gaussian", "Segment", "Square"))
    elif args.metric == "deletion":
        deletion_names = [f"masker_constant.deletion", "masker_blur.deletion", "masker_random.deletion"]
        wilcoxon_tests(res_obj, "Deletion", deletion_names, args.mode, args.activation, args.effect_size_measure, ("Constant", "Blur", "Random"))