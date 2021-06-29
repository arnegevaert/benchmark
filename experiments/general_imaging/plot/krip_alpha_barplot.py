import argparse
import pandas as pd
from attrbench.lib import krippendorff_alpha
from attrbench.suite import SuiteResult
from experiments.general_imaging.plot.dfs import get_default_dfs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from os import path
import glob
import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_file", type=str)
    parser.add_argument("-d", "--datasets", type=str, nargs="*")
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")
    sns.set()

    if args.datasets is None:
        result_files = glob.glob(path.join(args.in_dir, "*.h5"))
    else:
        result_files = [path.join(args.in_dir, f"{ds}.h5") for ds in args.datasets]
    result_objects = {path.basename(file).split(".")[0]: SuiteResult.load_hdf(file) for file in result_files}

    k_a = {
        ds_name: {
            metric_name: krippendorff_alpha(df.to_numpy())
            for metric_name, (df, _) in get_default_dfs(res, mode="single").items()
        } for ds_name, res in result_objects.items()
    }
    k_a = pd.DataFrame(k_a)

    k_a = k_a.reindex(["mnist", "fashionmnist", "cifar10", "cifar100", "svhn", "imagenet", "caltech", "places"], axis=1)
    color_list = [
        "#036d08",
        "#9de78c",
        "#08036d",
        "#845eb3",
        "#e7c3ff",
        "#6d012a",
        "#b65a73",
        "#ffaac4"
    ]

    color_dict = {
        "mnist": "#036d08",
        "fashionmnist": "#9de78c",
        "cifar10": "#08036d",
        "cifar100": "#845eb3",
        "svhn": "#e7c3ff",
        "imagenet": "#6d012a",
        "caltech": "#b65a73",
        "places": "#ffaac4"
    }

    fig, ax = plt.subplots(figsize=(16,8))
    k_a.plot.bar(ax=ax, color=color_list, width=0.7)
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.grid(axis="x")
    fig.savefig(args.out_file, bbox_inches="tight")