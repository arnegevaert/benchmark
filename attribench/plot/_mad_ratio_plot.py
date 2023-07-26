import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from attribench.plot import Plot


def mad_ratio(df):
    # Basically the F-ratio but using MAD instead of SS
    # MAD = median of absolute deviation to median
    # Mean of MADs for each method
    group_medians = df.median()
    within_mad = df.sub(group_medians).abs().median().mean()

    # MAD of group medians to global median
    global_median = df.stack().median()
    # between_mad = group_medians.sub(global_median).abs().median()
    between_mad = df.stack().sub(global_median).abs().median()

    return between_mad / within_mad


class MADRatioPlot(Plot):
    """
    Bar plot of the MAD (Median Absolute Deviation) ratio for each metric.
    The MAD ratio is the ratio of the total MAD to the MAD within groups.
    If this ratio is high, it means that there are differences in method
    behaviour.
    If the ratio is close to 1, it means that the methods behave similarly.
    This can be viewed as a one-way ANOVA test, but using non-parametric MAD
    instead of sum-of-squares.
    """

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
        ratios = {}
        for metric_name, (df, inverted) in self.dfs.items():
            ratios[metric_name] = mad_ratio(df)
        df = pd.DataFrame(ratios, index=["MAD Ratio"]).transpose()
        fig, ax = plt.subplots()
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        df.plot.barh(figsize=figsize, ax=ax)
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()
        return fig
