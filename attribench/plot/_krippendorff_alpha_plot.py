import pandas as pd
from krippendorff import krippendorff
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from attribench.plot import Plot
from matplotlib.figure import Figure


class KrippendorffAlphaPlot(Plot):
    def render(self, title=None, figsize=(10, 10), fontsize=20) -> Figure:
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
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()
        return fig
