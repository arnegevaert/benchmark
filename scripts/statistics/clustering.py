import pandas as pd
from os import path
import matplotlib.pyplot as plt
from scripts.statistics.df_extractor import DFExtractor
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def clustering(dfe: DFExtractor, out_dir: str, baseline: str):
    dfs = dfe.get_dfs()

    avg_values = {}
    for metric_name, (df, inverted) in dfs.items():
        df = df[df.columns.difference([baseline])]
        avg_values[metric_name] = df.mean(axis=0) if not inverted else -df.mean(axis=0)

    df = pd.DataFrame.from_dict(avg_values)
    normalized = MinMaxScaler().fit_transform(df)
    normalized = pd.DataFrame(normalized, columns=df.columns, index=df.index)

    fig = sns.clustermap(normalized, figsize=(7, 7))
    plt.tight_layout()
    out_file = path.join(out_dir, "clustering.png")
    fig.savefig(out_file)
