from typing import Dict
import pandas as pd
from attrbench.suite.plot.lib import heatmap
from attrbench.lib.stat import wilcoxon_tests
import seaborn as sns


class WilcoxonSummaryPlot:
    def __init__(self, dfs: Dict[str, pd.DataFrame], inverted: Dict[str, bool]):
        self.dfs = dfs
        self.inverted = inverted

    def render(self, figsize=(20, 20), glyph_scale=1500, fontsize=None):
        pvalues, effect_sizes = {}, {}
        for metric_name, df in self.dfs.items():
            mes, mpv = wilcoxon_tests(df, self.inverted[metric_name])
            effect_sizes[metric_name] = mes
            pvalues[metric_name] = mpv
        pvalues = pd.DataFrame(pvalues)
        effect_sizes = pd.DataFrame(effect_sizes).abs()
        effect_sizes = (effect_sizes - effect_sizes.min()) / (effect_sizes.max() - effect_sizes.min())
        effect_sizes[pvalues > 0.01] = 0

        effect_sizes = pd.melt(effect_sizes.reset_index(), id_vars='index')
        effect_sizes.columns = ["method", "metric", "value"]

        return heatmap(
            x=effect_sizes["method"],
            y=effect_sizes["metric"],
            size=effect_sizes["value"],
            color=effect_sizes["value"],
            palette=sns.color_palette("rocket_r", n_colors=256),
            color_min=0, color_max=1,
            figsize=figsize,
            glyph_scale=glyph_scale,
            fontsize=fontsize
        )
