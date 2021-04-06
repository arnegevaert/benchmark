import pandas as pd
import matplotlib.pyplot as plt
from scripts.statistics.util import krippendorff_alpha
from scripts.statistics.df_extractor import DFExtractor


def krippendorff(dfe: DFExtractor, baseline: str, out_file: str):
    dfs = dfe.get_dfs()
    k_a = {}
    for metric_name, (df, inverted) in dfs.items():
        df = df[df.columns.difference([baseline])]
        k_a[metric_name] = krippendorff_alpha(df.to_numpy())
    k_a = pd.DataFrame(k_a, index=["Krippendorff Alpha"]).transpose()
    fig, ax = plt.subplots()
    k_a.plot.barh(figsize=(10, 10), ax=ax)
    fig.tight_layout()
    fig.savefig(out_file)
