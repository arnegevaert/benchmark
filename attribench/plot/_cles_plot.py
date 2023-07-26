import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from attribench.plot import Plot


class CLESPlot(Plot):
    """Bar plot showing the CLES (Common Language Effect Size) of two methods.
    The CLES is the probability that the metric value for a random sample from
    one method will be greater than a random sample from the other method.

    This plot shows the CLES as a bar vertical bar plot centered around 0.5
    (no difference between methods). If the bar is on the right side of 0.5,
    it means that the first method is better than the second method. A bar
    is shown for each metric.
    """

    def render(self, method1: str, method2: str) -> Figure:
        """Render the plot.
        TODO add options for figsize, fontsize, title

        Parameters
        ----------
        method1 : str
            First method to compare.
        method2 : str
            Second method to compare.

        Returns
        -------
        Figure
            Rendered Matplotlib figure.

        Raises
        ------
        ValueError
            If one of the methods is not found in the dataframe.
        """
        result_cles = {}
        for key in self.dfs:
            df, higher_is_better = self.dfs[key]
            if not higher_is_better:
                df = -df

            if method1 not in df.columns:
                raise ValueError(
                    f"Method {method1} not found in dataframe {key}"
                )
            if method2 not in df.columns:
                raise ValueError(
                    f"Method {method2} not found in dataframe {key}"
                )

            statistic, pvalue = stats.wilcoxon(
                x=df[method1], y=df[method2], alternative="two-sided"
            )
            cles = (df[method1] > df[method2]).mean()
            result_cles[key] = cles if pvalue < 0.01 else 0.5
        sns.set_color_codes("muted")
        df = (
            pd.DataFrame(result_cles, index=["CLES"])
            .transpose()
            .reset_index()
            .rename(columns={"index": "Metric"})
        )
        df["CLES"] -= 0.5
        fig, ax = plt.subplots(figsize=(5, 7))
        sns.barplot(data=df, x="CLES", y="Metric", color="b", left=0.5, ax=ax)
        ax.set(xlim=(0, 1))
        ax.set_title(f"P({method1} > {method2})")
        return fig
