import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import path


def lineplot(exp_dir, x, y, xlog=False):
    df = pd.read_pickle(path.join(exp_dir, "result.pkl"))
    plt.figure(figsize=(7, 4))
    if xlog:
        plt.xscale("log")
    sns.lineplot(x=x, y=y, hue="method", data=df)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def boxplot(exp_dir, x, y, xtick_rotation):
    df = pd.read_pickle(path.join(exp_dir, "result.pkl"))
    plt.figure(figsize=(7, 4))
    ax = sns.boxplot(x=x, y=y, data=df, showfliers=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
