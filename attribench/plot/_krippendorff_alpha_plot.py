import pandas as pd
from krippendorff import krippendorff
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from attribench.plot import Plot
from matplotlib.figure import Figure


class KrippendorffAlphaPlot(Plot):
    """Bar plot of Krippendorff's alpha for each metric."""

    def render(
        self, title: str | None = None, figsize=(10, 10), fontsize=20
    ) -> Figure:
        """Render the plot.

        Parameters
        ----------
        title : str | None, optional
            Title of the figure, by default None
        figsize : tuple, optional
            Size of the figure, by default (10, 10)
        fontsize : int, optional
            Font size of x and y axis ticks, by default 20

        Returns
        -------
        Figure
            Rendered Matplotlib figure.
        """
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
