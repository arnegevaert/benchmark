import pandas as pd
from tqdm import tqdm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


class ConvergencePlot:
    """
    Line plot of the median values of the metrics vs. the number of samples.
    Error bars are computed using bootstrapping.
    Allows the user to inspect if metric values have converged,
    i.e. if the benchmark has been run on enough samples.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def render(self, title=None, bs_samples=1000, interval=5):
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
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.lineplot(
            data=all_medians,
            x="bs_size",
            y="value",
            hue="method",
            estimator=np.median,
            errorbar="ci",
            ax=ax,
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        ax.set_title(title)
        return fig
