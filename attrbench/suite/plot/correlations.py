import pandas as pd
from typing import Dict
from scipy.stats import pearsonr
import numpy as np
from attrbench.suite.plot.lib import heatmap


def _corr_heatmap(df, figsize=(20, 20), glyph_scale=1500, fontsize=None):
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
        fontsize=fontsize
    )
    return fig


class InterMetricCorrelationPlot:
    def __init__(self, dfs: Dict[str, pd.DataFrame], inverted: Dict[str, bool]):
        self.dfs = dfs
        self.inverted = inverted

    def render(self, figsize=(20, 20), glyph_scale=1500, fontsize=None):
        flattened_dfs = {}
        for metric_name, df in self.dfs.items():
            df = (df - df.min()) / (df.max() - df.min())
            all_columns = [df[column].to_numpy() for column in sorted(df.columns)]
            flattened_dfs[metric_name] = np.concatenate(all_columns)
        df = pd.DataFrame(flattened_dfs)
        df = df.reindex(sorted(df.columns), axis=1)
        return _corr_heatmap(df, figsize, glyph_scale, fontsize)


class InterMethodCorrelationPlot:
    def __init__(self, dfs: Dict[str, pd.DataFrame], inverted: Dict[str, bool]):
        self.dfs = dfs
        self.inverted = inverted

    def render(self, return_individual_figs=False, figsize=(20, 20), glyph_scale=1500, fontsize=None):
        all_dfs = [value if not self.inverted[key] else -value for key, value in self.dfs.items()]
        fig = _corr_heatmap(pd.concat(all_dfs))
        if return_individual_figs:
            individual_metric_figs = [_corr_heatmap(df, figsize, glyph_scale, fontsize) for df in all_dfs]
            return fig, individual_metric_figs
        return fig
