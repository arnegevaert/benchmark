import argparse
import pandas as pd
import matplotlib.pyplot as plt
from attrbench.suite import SuiteResult
from os import path
import os
import logging
from scripts.statistics.util import get_dfs
import seaborn as sns
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    # Constant parameters, might be moved to args if necessary
    IGNORE_METHODS = ["Random_pos_only", "GradCAM_no_relu", "GuidedGradCAM_no_relu"]
    BASELINE = "Random"
    PWR_ITERATIONS = 1000
    ALPHA = 0.01
    plt.rcParams["figure.dpi"] = 140
    res_obj = SuiteResult.load_hdf(args.file)

    if not path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S")

    inter_metric_dir = path.join(args.out_dir, "inter_metric_correlations")
    inter_method_dir = path.join(args.out_dir, "inter_method_correlations")
    for dir in (inter_metric_dir, inter_method_dir):
        if not path.isdir(dir):
            os.makedirs(dir)

    # Inter-method correlations
    all_dfs = []
    for metric_name in tqdm(res_obj.metric_results.keys()):
        if "until_flip" not in metric_name:  # TODO handle del_until_flip as well
            dfs = get_dfs(res_obj, metric_name, BASELINE, IGNORE_METHODS)
            for variant, (df, baseline, inverted) in dfs.items():
                """
                corrs = df.corr(method="spearman")
                fig, ax = plt.subplots(figsize=(10, 10))
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                plot = sns.heatmap(corrs, vmin=-1, vmax=1, center=0, ax=ax, cmap=cmap)
                fig.tight_layout()
                fig.savefig(path.join(inter_method_dir, f"{metric_name}_{variant}.png"))
                plt.close(fig)
                """
                all_dfs.append((df - df.min())/(df.max() - df.min()))

    corrs = pd.concat(all_dfs).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    plot = sns.heatmap(corrs, vmin=-1, vmax=1, center=0, ax=ax, cmap=cmap)
    fig.tight_layout()
    fig.savefig(path.join(args.out_dir, "inter_method_correlation.png"))
    plt.close(fig)

    # Inter-metric correlations
    flattened_dfs = {}
    for metric_name in tqdm(res_obj.metric_results.keys()):
        if "until_flip" not in metric_name:  # TODO handle del_until_flip as well
            dfs = get_dfs(res_obj, metric_name, BASELINE, IGNORE_METHODS)
            for variant, (df, baseline, inverted) in dfs.items():
                all_columns = []
                df = (df - df.min()) / (df.max() - df.min())
                for column in sorted(df.columns):
                    all_columns.append(df[column])
                flattened_dfs[f"{metric_name}_{variant}"] = pd.concat(all_columns)
    corrs = pd.concat(flattened_dfs, axis=1).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    plot = sns.heatmap(corrs, vmin=-1, vmax=1, center=0, ax=ax, cmap=cmap)
    fig.tight_layout()
    fig.savefig(path.join(args.out_dir, "inter_metric_correlation.png"))
    plt.close(fig)
