import argparse
import pandas as pd
import matplotlib.pyplot as plt
from attrbench.suite import SuiteResult
from os import path
import os
import logging
from scripts.statistics.df_extractor import DFExtractor
from scripts.statistics.util import correlation_heatmap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    # Constant parameters, might be moved to args if necessary
    EXCLUDE = ["Random_pos_only", "GradCAM_no_relu", "GuidedGradCAM_no_relu"]
    plt.rcParams["figure.dpi"] = 140
    RES_OBJ = SuiteResult.load_hdf(args.file)

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

    dfe = DFExtractor(RES_OBJ, EXCLUDE)
    dfe.add_infidelity("mse", "linear")
    dfe.add_infidelity("corr", "linear")
    dfe.compare_maskers(["constant", "blur", "random"], "linear")
    dfs = dfe.get_dfs()

    # Inter-method correlations
    normalized_dfs = []
    for metric_name, (df, inverted) in dfs.items():
        correlation_heatmap(df, path.join(inter_method_dir, f"{metric_name}.png"))
        normalized = (df - df.min())/(df.max() - df.min())
        normalized_dfs.append(normalized if not inverted else -normalized)
    correlation_heatmap(pd.concat(normalized_dfs), path.join(args.out_dir, "inter_method_correlation.png"))

    # Inter-metric correlations
    flattened_dfs = {}
    for metric_name, (df, inverted) in dfs.items():
        df = (df - df.min()) / (df.max() - df.min())
        all_columns = [df[column] for column in sorted(df.columns)]
        flattened_dfs[metric_name] = pd.concat(all_columns)
    df = pd.concat(flattened_dfs, axis=1)
    df = df.reindex(sorted(df.columns), axis=1)
    correlation_heatmap(df, path.join(args.out_dir, "inter_metric_correlation.png"))
