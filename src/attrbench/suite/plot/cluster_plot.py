from typing import Dict, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.cluster.hierarchy import linkage


class ClusterPlot:
    def __init__(self, dfs: Dict[str, Tuple[pd.DataFrame, bool]]):
        self.dfs = dfs

    def render(self, figsize=(7, 7)):
        medians = {}
        for metric_name, (df, inverted) in self.dfs.items():
            medians[metric_name] = df.median(axis=0) if not inverted else -df.median(axis=0)

        df = pd.DataFrame.from_dict(medians)
        normalized = MinMaxScaler().fit_transform(df)
        normalized_df = pd.DataFrame(normalized, columns=df.columns, index=df.index)

        # Manually computing the linkage so that we can set optimal_ordering to True
        # This should allow for easy comparison between clustermaps
        row_linkage = linkage(normalized, method="single", metric="correlation", optimal_ordering=True)
        col_linkage = linkage(np.transpose(normalized), method="single", metric="correlation", optimal_ordering=True)
        fig = sns.clustermap(normalized_df, row_linkage=row_linkage, col_linkage=col_linkage,
                             figsize=figsize).fig

        #fig = sns.clustermap(normalized, figsize=figsize, metric="correlation", method="single").fig
        #plt.tight_layout()
        return fig
