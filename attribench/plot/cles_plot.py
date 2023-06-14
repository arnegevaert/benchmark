import pandas as pd
from typing import Dict, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class CLESPlot:
    def __init__(self, dfs: Dict[str, Tuple[pd.DataFrame, bool]]):
        self.dfs = dfs

    def render(self, method1, method2):
        result_cles = {}
        for key in self.dfs:
            df, higher_is_better = self.dfs[key]
            if not higher_is_better:
                df = -df

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