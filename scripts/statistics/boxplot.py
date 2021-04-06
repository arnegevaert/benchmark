from scripts.statistics.df_extractor import DFExtractor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def boxplot(dfe: DFExtractor, out_file: str):
    dfs = dfe.get_dfs()
    df_list = []
    for name, (df, inverted) in dfs.items():
        df = df.melt()
        df["metric_name"] = name
        df_list.append(df)
    full_df = pd.concat(df_list)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.tight_layout()
    sns.boxplot(x="value", y="variable", hue="metric_name",
                data=full_df, orient="h", ax=ax)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
