import pandas as pd
import argparse
from os import path
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    mpl.use("Agg")
    sns.set()

    for filename in ("single", "batch"):
        df = np.log(pd.read_csv(path.join(args.in_dir, f"{filename}.csv"), index_col=0))
        fig, ax = plt.subplots(figsize=(16,8))
        df.plot.box(ax=ax)
        plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
        fig.savefig(path.join(args.out_dir, f"{filename}.png"), bbox_inches="tight")
        plt.close(fig)
