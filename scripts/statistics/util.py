import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def get_df(res, name, variant=None, activation=None, ignore_methods=None, log=False):
    # Get variant/activation
    if variant is None and activation is None:
        df = res.metric_results[name].to_df()
    elif variant is not None and activation is not None:
        df = res.metric_results[name].to_df()[f"{variant}_{activation}"]
    elif variant is not None:
        df = res.metric_results[name].to_df()[f"{variant}"]
    elif activation is not None:
        df = res.metric_results[name].to_df()[f"{activation}"]
    
    # Ignore columns
    df = df[df.columns.difference(ignore_methods)]
    
    # Log-transform
    if log:
        df = np.log(df)
    return df


def boxplot_grid(res, names, variant, activations, ignore_methods=None, log=False):
    dfs = []
    for activation in activations:
        for name in names:
            dfs.append((f"{name} - {activation}", get_df(res, name, variant, activation, ignore_methods, log)))

    fig, axs = plt.subplots(nrows=len(activations), ncols=len(names), figsize=(10*len(names), 10*len(activations)))
    axs = axs.flatten()
    for i, (title, df) in enumerate(dfs):
        sns.boxplot(ax=axs[i], data=df, orient="h")
        axs[i].set_title(title)