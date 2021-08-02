import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from scipy.stats import pearsonr
import numpy as np
from attrbench.suite.plot.lib import heatmap
import seaborn as sns


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
        self.methods = list(dfs.values())[0][0].columns

    def render(self, figsize=(20, 20), glyph_scale=1500, fontsize=12):
        corr_dfs = []
        for method_name in self.methods:
            data = {}
            for metric_name, (df, inverted) in self.dfs.items():
                data[metric_name] = -df[method_name].to_numpy() if inverted else df[method_name].to_numpy()
            df = pd.DataFrame(data)
            corr_dfs.append(df.corr(method="spearman"))
        corr = pd.concat(corr_dfs).mean(level=0)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, annot=True, vmin=-1, vmax=1, cmap=sns.diverging_palette(220, 20, as_cmap=True), ax=ax)
        ax.set_aspect("equal")
        return fig

        """
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
            color_bounds=(-1, 1)
        )
        return fig
        """


class InterMethodCorrelationPlot:
    def __init__(self, dfs: Dict[str, Tuple[pd.DataFrame, bool]]):
        self.dfs = dfs

    def render(self, figsize=(20, 20), glyph_scale=1500, fontsize=None):
        # Compute correlations for each metric
        all_dfs = [df if not inverted else -df for _, (df, inverted) in self.dfs.items()]
        corr_dfs = [df.corr(method="pearson") for df in all_dfs]

        # Compute average of correlations
        corr = pd.concat(corr_dfs).mean(level=0)
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
            color_bounds=(-1, 1)
        )
        return fig

    def render_all(self, figsize=(20, 20), glyph_scale=1500, fontsize=None):
        figs = {}
        for name, (df, inverted) in self.dfs.items():
            if inverted:
                df = -df
            corr = df.corr(method="pearson")
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
                color_bounds=(-1, 1)
            )
            figs[name] = fig
        return figs
