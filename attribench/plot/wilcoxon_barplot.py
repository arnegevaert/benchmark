from typing import Dict, Tuple
import pandas as pd
from attribench.plot.lib import effect_size_barplot
from attribench.stat import wilcoxon_tests


class WilcoxonBarPlot:
    """
    Alternative to WilcoxonSummaryPlot.
    Shows a bar plot of effect sizes, along with a grid showing statistical
    significance (green if significant, red otherwise).
    Provides more detailed information about effect size than the summary plot.
    Can get crowded if there are many methods/metrics.
    """

    def __init__(self, dfs: Dict[str, Tuple[pd.DataFrame, bool]]):
        self.dfs = dfs

    def render(self, figsize=(12, 6)):
        ALPHA = 0.01
        effect_sizes, pvalues = {}, {}

        for metric_name, (df, higher_is_better) in self.dfs.items():
            # Compute effect sizes and p-values
            mes, mpv = wilcoxon_tests(df, higher_is_better)
            effect_sizes[metric_name] = mes
            pvalues[metric_name] = mpv

        es_df = pd.DataFrame.from_dict(effect_sizes)
        pv_df = pd.DataFrame.from_dict(pvalues)
        fig, axs = effect_size_barplot(
            es_df, pv_df, self.dfs.keys(), ALPHA, figsize=figsize
        )
        return fig
