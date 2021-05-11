import pandas as pd
from attrbench.lib import krippendorff_alpha
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm


class KrippendorffAlphaPlot:
    def __init__(self, dfs: Dict[str, Tuple[pd.DataFrame, bool]]):
        self.dfs = dfs

    def render(self, title=None, fontsize=20, figsize=(10, 10)):
        k_a = {metric_name: krippendorff_alpha(df.to_numpy()) for metric_name, (df, _) in self.dfs.items()}
        k_a = pd.DataFrame(k_a, index=["Krippendorff Alpha"]).transpose()
        fig, ax = plt.subplots()
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        k_a.plot.barh(figsize=figsize, ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        return fig


class KrippendorffAlphaBootstrapPlot:
    def __init__(self, dfs: Dict[str, Tuple[pd.DataFrame, bool]]):
        self.dfs = dfs

    def render(self, title=None, bs_samples=100, min=1, max=50, step=5):
        data = {}
        x_range = list(range(min, max, step))
        for metric_name, (df, inverted) in self.dfs.items():
            data[metric_name] = [
                krippendorff_alpha(
                    pd.DataFrame(
                        [df.sample(n=bs_size, replace=True).median(axis=0) for _ in range(bs_samples)]).to_numpy())
                for bs_size in range(min, max, step)
            ]
        df = pd.DataFrame(data, index=x_range)
        fig, ax = plt.subplots()
        df.plot.line(ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        return fig
