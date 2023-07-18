import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.cluster.hierarchy import linkage
from attribench.plot import Plot
from matplotlib.figure import Figure


class ClusterPlot(Plot):
    """
    Clustermap of the median values of the metrics and methods.
    Allows the user to see which metrics and/or methods behave similarly.

    The plot is shown as a heatmap, with each cell corresponding to the median
    metric value for a given method and metric. The heatmap is clustered using
    hierarchical clustering, with the distance between two methods being the
    correlation between their median metric values. The distance between two
    metrics is computed in the same way. The clustering is performed using
    single linkage.
    """
    def render(self, figsize=(7, 7)) -> Figure:
        """Render the plot.
        TODO add more parameters for font size, title, etc.

        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Figure size, by default (7, 7)
        """
        medians = {}
        for metric_name, (df, higher_is_better) in self.dfs.items():
            medians[metric_name] = (
                df.median(axis=0) if higher_is_better else -df.median(axis=0)
            )

        df = pd.DataFrame.from_dict(medians)
        normalized = MinMaxScaler().fit_transform(df)
        normalized_df = pd.DataFrame(
            normalized, columns=df.columns, index=df.index
        )

        # Manually computing the linkage so that we can set 
        # optimal_ordering to True
        # This should allow for easy comparison between clustermaps
        row_linkage = linkage(
            normalized,
            method="single",
            metric="correlation",
            optimal_ordering=True,
        )
        col_linkage = linkage(
            np.transpose(normalized),
            method="single",
            metric="correlation",
            optimal_ordering=True,
        )
        fig = sns.clustermap(
            normalized_df,
            row_linkage=row_linkage,
            col_linkage=col_linkage,
            figsize=figsize,
        ).fig
        return fig
