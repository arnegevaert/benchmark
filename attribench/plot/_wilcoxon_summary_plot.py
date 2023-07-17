from typing import Tuple, List
import pandas as pd
from attribench.plot._lib import heatmap
from attribench.plot import Plot
from attribench._stat import wilcoxon_tests
from matplotlib.figure import Figure
import seaborn as sns


class WilcoxonSummaryPlot(Plot):
    """Summary plot for Wilcoxon tests.
    Plots the significance and effect sizes of Wilcoxon tests for each metric
    and each method in a heatmap. Every glyph on the heatmap corresponds to a
    Wilcoxon test between a given method and the random baseline on the given
    metric.
    """
    def render(
        self,
        title: str | None = None,
        figsize: Tuple[int, int] = (20, 20),
        fontsize: int | None = None,
        glyph_scale: int = 1500,
        method_order: List[str] | None =None,
    ) -> Figure:
        """Render the plot.

        Parameters
        ----------
        title : str | None, optional
            Title of the figure, by default None
        figsize : Tuple[int, int], optional
            Figure size, by default (20, 20)
        fontsize : int | None, optional
            Font size of x and y axis ticks, by default None
        glyph_scale : int, optional
            Scale of heatmap glyphs, by default 1500
        method_order : List[str] | None, optional
            Order in which methods should be displayed.
            If None, the order of the keys in the dictionary that was passed to
            the constructor will be used.
            By default None.

        Returns
        -------
        Figure
            The rendered Matplotlib figure.
        """
        pvalues, effect_sizes = {}, {}
        for metric_name, (df, higher_is_better) in self.dfs.items():
            mes, mpv = wilcoxon_tests(df, higher_is_better)
            effect_sizes[metric_name] = mes
            pvalues[metric_name] = mpv
        pvalues = pd.DataFrame(pvalues)
        effect_sizes = pd.DataFrame(effect_sizes).abs()
        effect_sizes[pvalues > 0.01] = 0

        # Normalize each column of the effect sizes dataframe
        effect_sizes = effect_sizes / effect_sizes.max()
        effect_sizes = effect_sizes.fillna(0)

        effect_sizes = pd.melt(effect_sizes.reset_index(), id_vars=["index"])
        effect_sizes.columns = ["method", "metric", "value"]

        return heatmap(
            x=effect_sizes["method"],
            y=effect_sizes["metric"],
            size=effect_sizes["value"],
            color=effect_sizes["value"],
            palette=sns.color_palette("Greens", n_colors=256),
            figsize=figsize,
            glyph_scale=glyph_scale,
            fontsize=fontsize,
            title=title,
            cbar=False,
            x_labels=method_order,
        )
