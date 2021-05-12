import pandas as pd
from typing import Dict, Tuple
from scipy.stats import pearsonr
import numpy as np
from attrbench.suite.plot.lib import heatmap


def _corr_heatmap(df, figsize=(20, 20), glyph_scale=1500, fontsize=None, title=None):
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
        color=corr['value'],
        figsize=figsize, glyph_scale=glyph_scale,
        fontsize=fontsize,
        title=title,
        color_bounds=(-1, 1)
    )
    return fig


class InterMetricCorrelationPlot:
    def __init__(self, dfs: Dict[str, Tuple[pd.DataFrame, bool]]):
        self.dfs = dfs

    def render(self, return_individual_figs=False, figsize=(20, 20), glyph_scale=1500, fontsize=None):
        flattened_dfs = {}
        methods = None
        for metric_name, (df, inverted) in self.dfs.items():
            methods = sorted(df.columns)
            df = (df - df.min()) / (df.max() - df.min())
            df = -df if inverted else df
            all_columns = [df[column].to_numpy() for column in methods]
            flattened_dfs[metric_name] = np.concatenate(all_columns)
        df = pd.DataFrame(flattened_dfs)
        df = df.reindex(sorted(df.columns), axis=1)
        fig = _corr_heatmap(df, figsize, glyph_scale, fontsize, title="All methods")

        if return_individual_figs:
            individual_figs = {}
            for method_name in methods:
                method_dfs = {}
                for metric_name, (df, inverted) in self.dfs.items():
                    df = (df - df.min()) / (df.max() - df.min())
                    df = -df if inverted else df
                    method_dfs[metric_name] = df[method_name].to_numpy()
                df = pd.DataFrame(method_dfs)
                df = df.reindex(sorted(df.columns), axis=1)
                individual_figs[method_name] = _corr_heatmap(df, figsize, glyph_scale, fontsize, title=method_name)
            return fig, individual_figs
        return fig


class InterMethodCorrelationPlot:
    def __init__(self, dfs: Dict[str, Tuple[pd.DataFrame, bool]]):
        self.dfs = dfs

    def render(self, return_individual_figs=False, figsize=(20, 20), glyph_scale=1500, fontsize=None):
        all_dfs = [df if not inverted else -df for _, (df, inverted) in self.dfs.items()]
        fig = _corr_heatmap(pd.concat(all_dfs), figsize, glyph_scale, fontsize, title="All metrics")
        if return_individual_figs:
            individual_metric_figs = {
                name: _corr_heatmap(df, figsize, glyph_scale, fontsize, title=name) if not inverted
                else _corr_heatmap(-df, figsize, glyph_scale, fontsize, title=name)
                for name, (df, inverted) in self.dfs.items()
            }
            return fig, individual_metric_figs
        return fig
