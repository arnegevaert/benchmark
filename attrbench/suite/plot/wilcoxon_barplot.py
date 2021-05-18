from typing import Dict, Tuple
import pandas as pd
from attrbench.suite.plot.lib import effect_size_barplot
from attrbench.lib.stat import wilcoxon_tests


class WilcoxonBarPlot:
    def __init__(self, dfs: Dict[str, Tuple[pd.DataFrame, bool]]):
        self.dfs = dfs

    def render(self):
        ALPHA = 0.01
        effect_sizes, pvalues = {}, {}

        for metric_name, (df, inverted) in self.dfs.items():
            # Compute effect sizes and p-values
            mes, mpv = wilcoxon_tests(df, inverted)
            effect_sizes[metric_name] = mes
            pvalues[metric_name] = mpv

        es_df = pd.DataFrame.from_dict(effect_sizes)
        pv_df = pd.DataFrame.from_dict(pvalues)
        fig, axs = effect_size_barplot(es_df, pv_df, self.dfs.keys(), ALPHA)
        return fig
