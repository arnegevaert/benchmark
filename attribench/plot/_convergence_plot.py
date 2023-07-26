import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from attribench.plot import Plot
from matplotlib.figure import Figure


class ConvergencePlot(Plot):
    """ Line plot of the median values of a given metric vs the number of samples.
    Error bars are computed using bootstrapping.
    Allows the user to inspect if metric values have converged,
    i.e. if the benchmark has been run on enough samples.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Pandas dataframe containing the metric values.
            The columns are the names of the methods.
        """
        super().__init__({})
        self.df = df

    def render(
        self,
        title: str | None = None,
        figsize=(20, 20),
        fontsize=None,
        bs_samples=1000,
        interval=5,
    ) -> Figure:
        """Render the plot.

        Parameters
        ----------
        title : str | None, optional
            Title of the figure, by default None
        figsize : Tuple[int, int], optional
            Figure size, by default (20, 20)
        fontsize : int | None, optional
            Font size of x and y axis ticks, by default None
        bs_samples : int, optional
            Number of bootstrap samples for estimating the median value of the
            metric using a given sample size. By default 1000
        interval : int, optional
            Interval between sample sizes, by default 5

        Returns
        -------
        Figure
            The rendered Matplotlib figure.
        """
        all_medians = []
        for bs_size in tqdm(range(2, self.df.shape[0], interval)):
            medians = []
            for _ in range(bs_samples):
                sample = self.df.sample(n=bs_size, replace=True)
                medians.append(sample.median(axis=0))  # median for each column
            medians = pd.DataFrame(medians)

            medians = pd.melt(medians, var_name="method")
            medians["bs_size"] = bs_size

            all_medians.append(medians)
        all_medians = pd.concat(all_medians)
        fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(
            data=all_medians,
            x="bs_size",
            y="value",
            hue="method",
            estimator="median",
            errorbar="ci",
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), size=fontsize)
        ax.set_yticklabels(ax.get_yticklabels(), size=fontsize)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        if title is not None:
            ax.set_title(title)
        return fig
