import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.statistics.util import cohend, emp_power_curve, wilcoxon_tests
from scripts.statistics.plot import plot_wilcoxon_result
from scripts.statistics.df_extractor import DFExtractor
import os
from os import path


def wilcoxon(dfe: DFExtractor, es_measure: str, baseline: str, out_file: str, power_curves=False, power_dir=None):
    ALPHA = 0.01
    # variant -> (metric_name -> (method_name -> es/pv))
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
