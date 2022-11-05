import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import seaborn as sns


def _create_fig(df, figsize, annot):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=annot, vmin=-1, vmax=1, cmap=sns.diverging_palette(220, 20, as_cmap=True), ax=ax, fmt=".2f",
                cbar=False)
    ax.set_aspect("equal")
    return fig


class InterMetricCorrelationPlot:
    def __init__(self, dfs: Dict[str, Tuple[pd.DataFrame, bool]]):
        self.dfs = dfs
        self.methods = list(dfs.values())[0][0].columns

    def render(self, figsize=(20, 20), annot=True):
        corr_dfs = []
        for method_name in self.methods:
            data = {}
            for metric_name, (df, inverted) in self.dfs.items():
                data[metric_name] = -df[method_name].to_numpy() if inverted else df[method_name].to_numpy()
            df = pd.DataFrame(data)
            corr_dfs.append(df.corr(method="spearman"))
        corr = pd.concat(corr_dfs).mean(level=0)
        return _create_fig(corr, figsize, annot)


class InterMethodCorrelationPlot:
    def __init__(self, dfs: Dict[str, Tuple[pd.DataFrame, bool]]):
        self.dfs = dfs

    def render(self, figsize=(20, 20), annot=True):
        # Compute correlations for each metric
        all_dfs = [df if not inverted else -df for _, (df, inverted) in self.dfs.items()]
        corr_dfs = [df.corr(method="spearman") for df in all_dfs]

        # Compute average of correlations
        corr = pd.concat(corr_dfs).mean(level=0)
        return _create_fig(corr, figsize, annot)

    def render_all(self, figsize=(20, 20), annot=True):
        figs = {}
        for name, (df, inverted) in self.dfs.items():
            if inverted:
                df = -df
            corr = df.corr(method="spearman")
            figs[name] = _create_fig(corr, figsize, annot)
        return figs
