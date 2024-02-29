from typing import Tuple, List
import pandas as pd
from attribench.plot._lib import heatmap
from attribench.plot import Plot
from attribench._stat import significance_tests
from matplotlib.figure import Figure
import seaborn as sns


class SignificanceSummaryPlot(Plot):
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
        method_order: List[str] | None = None,
        alpha=0.01,
        test="wilcoxon",
        multiple_testing="bonferroni",
        normalize=False,
    ) -> Figure:
        """
        Renders a significance summary plot.

        Args:
            title (str, optional): Title of the plot. Defaults to None.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (20, 20).
            fontsize (int, optional): Font size. Defaults to None.
            glyph_scale (int, optional): Scale factor for the glyphs. Defaults to 1500.
            method_order (List[str], optional): Order of the methods. Defaults to None.
            alpha (float, optional): Significance level. Defaults to 0.01.
            test (str, optional): Statistical test to use. Defaults to "wilcoxon".
                Can be one of "wilcoxon", "t_test", or "sign_test".
            multiple_testing (str, optional): Method for multiple testing correction. Defaults to "bonferroni".
                Can be one of "bonferroni", "fdr_bh", or None.
            normalize (bool, optional): Whether to normalize the effect sizes. Defaults to False.
                If True, the effect sizes will be normalized to the range [0, 1] by dividing by the maximum effect size.

        Returns:
            Figure: The rendered significance summary plot.
        """

        pvalues, effect_sizes = {}, {}
        for metric_name, (df, higher_is_better) in self.dfs.items():
            mes, mpv = significance_tests(
                df,
                alpha,
                test=test,
                alternative="greater" if higher_is_better else "less",
                multiple_testing=multiple_testing,
            )
            effect_sizes[metric_name] = mes
            pvalues[metric_name] = mpv

        # Hide effect sizes that are not significant
        pvalues = pd.DataFrame(pvalues)
        # Scale effect sizes from [0.5, 1] to [0, 1]
        effect_sizes = pd.DataFrame(effect_sizes) - 0.5
        effect_sizes *= 2
        effect_sizes[pvalues > alpha] = 0
        effect_sizes[effect_sizes < 0] = 0

        if normalize:
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
