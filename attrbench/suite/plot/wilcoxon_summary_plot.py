from typing import Dict, Tuple
import pandas as pd
from attrbench.suite.plot.lib import heatmap
from attrbench.lib.stat import wilcoxon_tests
import seaborn as sns


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
        norm_effect_sizes = (effect_sizes - effect_sizes.min()) / (effect_sizes.max() - effect_sizes.min())
        effect_sizes[pvalues > 0.01] = 0

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
