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


class AvgInterMetricCorrelationPlot:
    """Heatmap showing Spearman correlations between metrics averaged across
    multiple datasets."""

    def __init__(self, dfs: Dict[str, Dict[str, Tuple[pd.DataFrame, bool]]]):
        """
        Parameters
        ----------
        dfs : Dict[str, Dict[str, Tuple[pd.DataFrame, bool]]]
            A dictionary mapping dataset names to dictionaries mapping metric
            names to tuples of dataframes and booleans. The boolean indicates
            whether higher values of the metric are better (``True``) or not
            (``False``). The dataframes should have the same columns, which are
            the names of the methods.
        """
        self.dfs = dfs

    def render(
        self,
        title: str | None = None,
        figsize: Tuple[int, int] = (20, 20),
        fontsize: int | None = None,
        annot: bool = False,
    ) -> Figure:
        """Render the plot.

        Parameters
        ----------
        title : str | None, optional
            Title of the figure, by default None
        figsize : Tuple[int, int], optional
            Size of the figure, by default (20, 20)
        fontsize : int | None, optional
            Font size of x and y axis ticks, by default None
        annot : bool, optional
            Whether to annotate the heatmap with the correlation values, by
            default False

        Returns
        -------
        Figure
            The rendered Matplotlib figure.
        """
        dataset_corr_dfs = []
        for dataset_name in self.dfs.keys():
            # For each dataset, get correlations between metrics
            # for all methods and average them
            dataset_dfs = self.dfs[dataset_name]
            method_corr_dfs = []
            methods = list(dataset_dfs.values())[0][0].columns
            for method_name in methods:
                data = {}
                for metric_name, (df, higher_is_better) in dataset_dfs.items():
                    data[metric_name] = (
                        -df[method_name].to_numpy()
                        if not higher_is_better
                        else df[method_name].to_numpy()
                    )
                df = pd.DataFrame(data)
                method_corr_dfs.append(df.corr(method="spearman"))
            corr = pd.concat(method_corr_dfs).groupby(level=0).mean()
            corr = corr.reindex(corr.columns)
            dataset_corr_dfs.append(corr)

        # Average correlations across datasets
        corr = pd.concat(dataset_corr_dfs).groupby(level=0).mean()
        corr = corr.reindex(corr.columns)
    
        # Create figure
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


class InterMetricCorrelationPlot(Plot):
    """Heatmap showing Spearman correlations between metrics."""

    def render(
        self,
        title: str | None = None,
        figsize: Tuple[int, int] = (20, 20),
        fontsize: int | None = None,
        annot: bool = False,
    ) -> Figure:
        """Render the plot.

        Parameters
        ----------
        title : str | None, optional
            Title of the figure, by default None
        figsize : Tuple[int, int], optional
            Size of the figure, by default (20, 20)
        fontsize : int | None, optional
            Font size of x and y axis ticks, by default None
        annot : bool, optional
            Whether to annotate the heatmap with the correlation values, by
            default False

        Returns
        -------
        Figure
            The rendered Matplotlib figure.
        """
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


class InterMethodCorrelationPlot(Plot):
    """Heatmap showing Spearman correlations between methods."""

    def render(
        self,
        title: str | None = None,
        figsize=(20, 20),
        fontsize: int | None = None,
        annot=False,
    ) -> Figure:
        """Render the plot.
        Spearman correlation values are averaged across metrics.
        To plot inter-method correlations for each metric separately,
        use :meth:`render_all`.

        Parameters
        ----------
        title : str | None, optional
            Title of the figure, by default None
        figsize : Tuple[int, int], optional
            Size of the figure, by default (20, 20)
        fontsize : int | None, optional
            Font size of x and y axis ticks, by default None
        annot : bool, optional
            Whether to annotate the heatmap with the correlation values, by
            default False

        Returns
        -------
        Figure
            The rendered Matplotlib figure.
        """
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

    def render_all(
        self, figsize=(20, 20), fontsize: int | None = None, annot=False
    ) -> Dict[str, Figure]:
        """Render a separate heatmap for each metric.
        TODO test and make sure args are consistent with render.

        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Size of the figures, by default (20, 20)
        fontsize : int | None, optional
            Font size of x and y axis ticks, by default None
        annot : bool, optional
            Whether to annotate the heatmaps with the correlation values, by
            default False

        Returns
        -------
        Dict[str, Figure]
            Dictionary mapping metric names to rendered Matplotlib figures.
        """
        figs = {}
        for name, (df, inverted) in self.dfs.items():
            if inverted:
                df = -df
            corr = df.corr(method="spearman")
            figs[name] = _create_fig(corr, figsize, annot)
        return figs
