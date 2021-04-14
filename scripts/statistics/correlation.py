import pandas as pd
import matplotlib.pyplot as plt
from os import path
import os
from scripts.statistics.df_extractor import DFExtractor
from scripts.statistics.plot import heatmap
from scipy.stats import pearsonr
import numpy as np


def corr_heatmap(df):
    corr = df.corr(method=lambda x, y: pearsonr(x, y)[0])
    pvalues = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(corr.shape[0])
    corr[pvalues > 0.01] = 0

    corr = pd.melt(corr.reset_index(),
                   id_vars='index')  # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']
    fig = heatmap(
        x=corr['x'],
        y=corr['y'],
        size=corr['value'].abs(),
        color=corr['value']
    )
    return fig




def correlation(dfe: DFExtractor, out_dir: str, baseline: str):
    inter_method_dir = path.join(out_dir, "inter_method_correlations")
    if not path.isdir(inter_method_dir):
        os.makedirs(inter_method_dir)

    dfs = dfe.get_dfs()

    # Relative Inter-method correlations
    all_dfs = []
    for metric_name, (df, inverted) in dfs.items():
        df = df.sub(df[baseline], axis=0)
        df = df[df.columns.difference([baseline])]
        fig = corr_heatmap(df)
        fig.savefig(path.join(inter_method_dir, f"{metric_name}.png"))
        plt.close(fig)
        all_dfs.append(df if not inverted else -df)
    fig = corr_heatmap(pd.concat(all_dfs))
    fig.savefig(path.join(out_dir, "inter_method_correlation.png"), bbox_inches="tight")
    plt.close(fig)

    # Inter-method correlations
    all_dfs = []
    inter_method_dir = path.join(out_dir, "inter_method_correlations_abs")
    if not path.isdir(inter_method_dir):
        os.makedirs(inter_method_dir)
    for metric_name, (df, inverted) in dfs.items():
        fig = corr_heatmap(df)
        fig.savefig(path.join(inter_method_dir, f"{metric_name}.png"))
        plt.close(fig)
        all_dfs.append(df if not inverted else -df)
    fig = corr_heatmap(pd.concat(all_dfs))
    fig.savefig(path.join(out_dir, "inter_method_correlation_abs.png"), bbox_inches="tight")
    plt.close(fig)

    # Inter-metric correlations
    flattened_dfs = {}
    for metric_name, (df, inverted) in dfs.items():
        df = (df - df.min()) / (df.max() - df.min())
        all_columns = [df[column] for column in sorted(df.columns)]
        flattened_dfs[metric_name] = pd.concat(all_columns)
    df = pd.concat(flattened_dfs, axis=1)
    df = df.reindex(sorted(df.columns), axis=1)
    fig = corr_heatmap(df)
    fig.savefig(path.join(out_dir, "inter_metric_correlation.png"), bbox_inches="tight")
    plt.close(fig)
