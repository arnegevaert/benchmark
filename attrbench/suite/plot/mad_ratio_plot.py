from typing import Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt

def mad_ratio(df):
    # Basically the F-ratio but using MAD instead of SS
    # MAD = median of absolute deviation to median
    # Mean of MADs for each method
    group_medians = df.median()
    within_mad = df.sub(group_medians).abs().median().mean()

    # MAD of group medians to global median
    global_median = df.stack().median()
    between_mad = group_medians.sub(global_median).abs().median()

    return between_mad / within_mad


class MADRatioPlot:
    def __init__(self, dfs: Dict[str, Tuple[pd.DataFrame, bool]]):
        self.dfs = dfs

    def render(self, title=None, fontsize=20, figsize=(10, 10)):
        ratios = {}
        for metric_name, (df, inverted) in self.dfs.items():
            ratios[metric_name] = mad_ratio(df)
        df = pd.DataFrame(ratios, index=["MAD Ratio"]).transpose()
        fig, ax = plt.subplots()
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        df.plot.barh(figsize=figsize, ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        return fig
