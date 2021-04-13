import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scripts.statistics.util import cohend, emp_power_curve, wilcoxon_tests
from scripts.statistics.plot import plot_wilcoxon_result, heatmap
from scripts.statistics.df_extractor import DFExtractor
import os
from os import path
from typing import Dict


def wilcoxon(dfe: DFExtractor, es_measure: str, baseline: str, out_file: str, power_curves=False, power_dir=None):
    ALPHA = 0.01
    effect_sizes, pvalues = {}, {}
    dfs = dfe.get_dfs()

    for metric_name, (df, inverted) in dfs.items():
        # Compute effect sizes and p-values
        baseline_array = df[[baseline]].to_numpy().flatten()
        df = df[df.columns.difference([baseline])]
        mes, mpv = wilcoxon_tests(df, baseline_array, es_measure, inverted)
        effect_sizes[metric_name] = mes
        pvalues[metric_name] = mpv

        # Compute power curves for significant methods
        if power_curves:
            PWR_ITERATIONS = 1000
            PWR_N_RANGE = np.arange(20, 250, 20)
            PWR_TOLERANCE = 0.8
            power_curves = {}
            for method_name in mpv.keys():
                if mpv[method_name] < ALPHA:
                    effect_size = cohend(df[method_name].to_numpy(), baseline_array)
                    power_curves[method_name] = emp_power_curve(df[method_name].to_numpy(),
                                                                baseline_array,
                                                                effect_size, PWR_ITERATIONS, PWR_N_RANGE, inverted,
                                                                PWR_TOLERANCE,
                                                                ALPHA)
            fig, ax = plt.subplots()
            for method_name, power_curve in power_curves.items():
                ax.plot(PWR_N_RANGE, power_curve, label=method_name)
            fig.legend()
            metric_out_dir = path.join(power_dir, metric_name)
            if not path.isdir(metric_out_dir):
                os.makedirs(metric_out_dir)
            fig.savefig(path.join(metric_out_dir, f"{metric_name.replace('.', '_')}_power.png"))

    es_df = pd.DataFrame.from_dict(effect_sizes)
    pv_df = pd.DataFrame.from_dict(pvalues)
    fig, axs = plot_wilcoxon_result(es_df, pv_df, dfs.keys(), ALPHA)
    fig.savefig(out_file)
    plt.close(fig)


def wilcoxon_summary(dfe: DFExtractor, es_measures: Dict[str, str], baseline: str, out_file: str = None):
    pvalues, effect_sizes = {}, {}
    dfs = dfe.get_dfs()

    for metric_name, (df, inverted) in dfs.items():
        baseline_array = df[[baseline]].to_numpy().flatten()
        df = df[df.columns.difference([baseline])]
        mes, mpv = wilcoxon_tests(df, baseline_array, es_measures[metric_name], inverted)
        effect_sizes[metric_name] = mes
        pvalues[metric_name] = mpv
    pvalues = pd.DataFrame(pvalues)
    effect_sizes = pd.DataFrame(effect_sizes).abs()
    effect_sizes = (effect_sizes - effect_sizes.min()) / (effect_sizes.max() - effect_sizes.min())
    effect_sizes[pvalues > 0.01] = 0

    effect_sizes = pd.melt(effect_sizes.reset_index(), id_vars='index')
    effect_sizes.columns = ["method", "metric", "value"]

    fig = heatmap(
        x=effect_sizes["method"],
        y=effect_sizes["metric"],
        size=effect_sizes["value"],
        color=effect_sizes["value"],
        palette=sns.color_palette("rocket_r", n_colors=256),
        color_min=0, color_max=1
    )
    if out_file:
        fig.savefig(out_file, bbox_inches="tight")


if __name__ == "__main__":
    from attrbench.suite import SuiteResult
    from collections import defaultdict

    RES_OBJ = SuiteResult.load_hdf("../../out/ImageNet_resnet18.h5")
    EXCLUDE = ["Random_pos_only", "GradCAM_no_relu", "GuidedGradCAM_no_relu", "EdgeDetection"]

    dfe = DFExtractor(RES_OBJ, EXCLUDE)
    dfe.add_infidelity("mse", "linear")
    dfe.add_infidelity("mse", "softmax")
    dfe.compare_maskers(["constant", "random", "blur"], "linear")
    wilcoxon_summary(dfe, defaultdict(lambda: "cohend"), "Random")
