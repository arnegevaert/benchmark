import pandas as pd
from tqdm import tqdm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


class ConvergencePlot:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def render(self, title=None):
        all_medians = []
        for bs_size in tqdm(range(2, len(self.df.index), 5)):
            medians = []
            for _ in range(1000):
                sample = self.df.sample(n=bs_size, replace=True)
                medians.append(sample.median(axis=0))  # median for each column
            medians = pd.DataFrame(medians)

            medians = pd.melt(medians, var_name="method")
            medians["bs_size"] = bs_size

            all_medians.append(medians)
        all_medians = pd.concat(all_medians)
        fig, ax = plt.subplots()
        sns.lineplot(data=all_medians, x="bs_size", y="value", hue="method", estimator=np.median, ci="sd", ax=ax)
        ax.set_title(title)
        return fig
