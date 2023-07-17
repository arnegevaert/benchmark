import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import seaborn as sns
from matplotlib.figure import Figure
from attribench.plot import Plot


def _create_fig(df, figsize, annot):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df,
        annot=annot,
        vmin=-1,
        vmax=1,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        ax=ax,
        fmt=".2f",
        cbar=False,
    )
    ax.set_aspect("equal")
    return fig, ax


class InterMetricCorrelationPlot(Plot):
    def render(
        self,
        title: str | None = None,
        figsize: Tuple[int, int] = (20, 20),
        fontsize: int | None = None,
        annot: bool = False,
    ) -> Figure:
        corr_dfs = []
        methods = list(self.dfs.values())[0][0].columns
        for method_name in methods:
            data = {}
            for metric_name, (df, higher_is_better) in self.dfs.items():
                data[metric_name] = (
                    -df[method_name].to_numpy()
                    if not higher_is_better
                    else df[method_name].to_numpy()
                )
            df = pd.DataFrame(data)
            corr_dfs.append(df.corr(method="spearman"))
        corr = pd.concat(corr_dfs).groupby(level=0).mean()
        corr = corr.reindex(corr.columns)
        fig, ax = _create_fig(corr, figsize, annot)
        if title is not None:
            ax.set_title(title)

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=fontsize,
        )
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
        return fig


class InterMethodCorrelationPlot:
    def __init__(self, dfs: Dict[str, Tuple[pd.DataFrame, bool]]):
        self.dfs = dfs

    def render(
        self,
        title: str | None = None,
        figsize=(20, 20),
        fontsize: int | None = None,
        annot=False,
    ) -> Figure:
        # Compute correlations for each metric
        all_dfs = [
            df if not inverted else -df
            for _, (df, inverted) in self.dfs.items()
        ]
        corr_dfs = [df.corr(method="spearman") for df in all_dfs]

        # Compute average of correlations
        corr = pd.concat(corr_dfs).groupby(level=0).mean()
        fig, ax = _create_fig(corr, figsize, annot)
        if title is not None:
            ax.set_title(title)
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=fontsize,
        )
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
        return fig

    def render_all(self, figsize=(20, 20), annot=True):
        figs = {}
        for name, (df, inverted) in self.dfs.items():
            if inverted:
                df = -df
            corr = df.corr(method="spearman")
            figs[name] = _create_fig(corr, figsize, annot)
        return figs
