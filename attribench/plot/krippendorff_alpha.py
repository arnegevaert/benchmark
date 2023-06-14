import pandas as pd
from krippendorff import krippendorff
from scipy.stats import rankdata
from typing import Dict, Tuple
import matplotlib.pyplot as plt


class KrippendorffAlphaPlot:
    def __init__(self, dfs: Dict[str, Tuple[pd.DataFrame, bool]]):
        self.dfs = dfs

    def render(self, title=None, fontsize=20, figsize=(10, 10)):
        k_a = {
            metric_name: krippendorff.alpha(
                rankdata(df.to_numpy(), axis=1), level_of_measurement="ordinal"
            )
            for metric_name, (df, _) in self.dfs.items()
        }
        k_a = pd.DataFrame(k_a, index=["Krippendorff Alpha"]).transpose()
        fig, ax = plt.subplots()
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        k_a.plot.barh(figsize=figsize, ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        return fig
