import pandas as pd
import matplotlib.pyplot as plt
from scripts.statistics.util import krippendorff_alpha, wilcoxon_tests
from scripts.statistics.df_extractor import DFExtractor


def krippendorff(dfe: DFExtractor, baseline: str, out_file: str = None, exclude_non_significant=False):
    dfs = dfe.get_dfs()
    k_a = {}
    for metric_name, (df, inverted) in dfs.items():
        if exclude_non_significant:
            _, pvalues = wilcoxon_tests(df[df.columns.difference([baseline])], df[[baseline]].to_numpy().flatten(), "cohend", inverted)
            exclude = [key for key in pvalues if pvalues[key] > 0.01]
            df = df[df.columns.difference(exclude)]
        df = df[df.columns.difference([baseline])]
        k_a[metric_name] = krippendorff_alpha(df.to_numpy())
    k_a = pd.DataFrame(k_a, index=["Krippendorff Alpha"]).transpose()
    fig, ax = plt.subplots()
    k_a.plot.barh(figsize=(10, 10), ax=ax)
    fig.tight_layout()
    if out_file:
        fig.savefig(out_file)
