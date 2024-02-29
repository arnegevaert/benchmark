from typing import Dict, Tuple
from numpy import typing as npt
import pandas as pd
from attribench._stat import significance_tests
from attribench.plot import Plot
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np


def _effect_size_barplot(
    effect_sizes, pvalues, labels, alpha, figsize, fontsize
) -> Tuple[Figure, npt.NDArray]:
    fig, axs = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [4, 2]})

    effect_sizes.plot.barh(figsize=figsize, ax=axs[0])
    axs[0].legend(
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        fancybox=True,
        shadow=True,
    )
    axs[1].pcolor(
        pvalues < alpha, cmap="RdYlGn", edgecolor="black", vmin=0.0, vmax=1.0
    )
    axs[1].set_title(f"p < {alpha}")
    axs[1].set_yticks([])
    axs[1].set_xticks(np.arange(len(labels)) + 0.5, fontsize=fontsize)

    axs[1].set_xticklabels(
        labels,
        ha="right",
        rotation=45,
        rotation_mode="anchor",
        fontsize=fontsize,
    )
    return fig, axs


class SignificanceBarPlot(Plot):
    """
    Alternative to WilcoxonSummaryPlot.
    Shows a bar plot of effect sizes, along with a grid showing statistical
    significance (green if significant, red otherwise).
    Provides more detailed information about effect size than the summary plot.
    Can get crowded if there are many methods/metrics.
    """

    def render(
        self,
        title: str | None = None,
        figsize=(12, 6),
        fontsize: int | None = None,
        alpha=0.01,
        test="wilcoxon",
        multiple_testing="bonferroni",
    ) -> Figure:
        """Render the plot.

        Parameters
        ----------
        title : str | None, optional
            Title of the figure, by default None
        figsize : tuple, optional
            Size of the figure, by default (12, 6)
        fontsize : int | None, optional
            Font size of x and y axis ticks, by default None

        Returns
        -------
        Figure
            Rendered Matplotlib figure.
        """
        effect_sizes, pvalues = {}, {}

        for metric_name, (df, higher_is_better) in self.dfs.items():
            # Compute effect sizes and p-values
            mes, mpv = significance_tests(
                df,
                alpha,
                test=test,
                alternative="greater" if higher_is_better else "less",
                multiple_testing=multiple_testing,
            )
            effect_sizes[metric_name] = mes
            pvalues[metric_name] = mpv

        es_df = pd.DataFrame.from_dict(effect_sizes)
        pv_df = pd.DataFrame.from_dict(pvalues)
        fig, axs = _effect_size_barplot(
            es_df,
            pv_df,
            self.dfs.keys(),
            alpha,
            figsize=figsize,
            fontsize=fontsize,
        )
        if title is not None:
            fig.suptitle(title)
        return fig
