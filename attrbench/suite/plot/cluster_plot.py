from typing import Dict, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class ClusterPlot:
    def __init__(self, dfs: Dict[str, Tuple[pd.DataFrame, bool]]):
        self.dfs = dfs

    def render(self, figsize=(7, 7)):
        medians = {}
        for metric_name, (df, inverted) in self.dfs.items():
            medians[metric_name] = df.median(axis=0) if not inverted else -df.median(axis=0)

        df = pd.DataFrame.from_dict(medians)
        normalized = MinMaxScaler().fit_transform(df)
        normalized = pd.DataFrame(normalized, columns=df.columns, index=df.index)
        fig = sns.clustermap(normalized, figsize=figsize, metric="correlation", method="complete").fig
        plt.tight_layout()
        return fig
