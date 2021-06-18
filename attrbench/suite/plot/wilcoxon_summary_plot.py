from typing import Dict, Tuple
import pandas as pd
from attrbench.suite.plot.lib import heatmap
from attrbench.lib.stat import wilcoxon_tests
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class WilcoxonSummaryPlot:
    def __init__(self, dfs: Dict[str, Tuple[pd.DataFrame, bool]]):
        self.dfs = dfs

    def render(self, figsize=(20, 20), glyph_scale=1500, fontsize=None, title=None):
        pvalues, effect_sizes = {}, {}
        for metric_name, (df, inverted) in self.dfs.items():
            mes, mpv = wilcoxon_tests(df, inverted)
            effect_sizes[metric_name] = mes
            pvalues[metric_name] = mpv
        pvalues = pd.DataFrame(pvalues)
        effect_sizes = pd.DataFrame(effect_sizes).abs()
        effect_sizes[pvalues > 0.01] = 0

        # Normalize each column of the effect sizes dataframe
        effect_sizes = (effect_sizes - effect_sizes.min()) / (effect_sizes.max() - effect_sizes.min())
        effect_sizes = effect_sizes.fillna(0)
        effect_sizes = effect_sizes.transpose()
        cbarlabel = "Normalized effect size"

        columns = list(effect_sizes.columns)
        rows = list(effect_sizes.index)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(np.ma.masked_where(effect_sizes == 0, effect_sizes), cmap="Greens")

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # Show all ticks
        ax.set_xticks(np.arange(len(columns)))
        ax.set_yticks(np.arange(len(rows)))
        # Label ticks
        ax.set_xticklabels(columns)
        ax.set_yticklabels(rows)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Turn spines off and create white grid.
        for key in ax.spines:
            ax.spines[key].set_visible(False)

        # Loop over data dimensions and create text annotations.
        for i in range(len(rows)):
            for j in range(len(columns)):
                es = effect_sizes.iloc[i, j]
                if es > 0:
                    color = "black" if es < 0.9 else "white"
                    text = ax.text(j, i, f"{es:.2f}",
                                   ha="center", va="center", color=color)
        return fig
        """
        effect_sizes = pd.melt(effect_sizes.reset_index(), id_vars='index')
        effect_sizes.columns = ["method", "metric", "value"]
        #norm_effect_sizes = pd.melt(norm_effect_sizes.reset_index(), id_vars='index')
        #norm_effect_sizes.columns = ["method", "metric", "value"]

        return heatmap(
            x=effect_sizes["method"],
            y=effect_sizes["metric"],
            size=effect_sizes["value"],
            color=effect_sizes["value"],
            palette=sns.color_palette("rocket_r", n_colors=256),
            figsize=figsize,
            glyph_scale=glyph_scale,
            fontsize=fontsize,
            title=title
        )
        """