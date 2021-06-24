import argparse
import pandas as pd
from attrbench.lib import krippendorff_alpha
from attrbench.suite import SuiteResult
from attrbench.suite.plot import InterMethodCorrelationPlot
from experiments.general_imaging.plot.dfs import get_default_dfs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from os import path
import os
import glob
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_file", type=str)
    parser.add_argument("-d", "--datasets", type=str, nargs="*")
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")

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

    fig, ax = plt.subplots(figsize=(10,10))
    k_a.plot.barh(ax=ax)
    plt.grid(axis="x")
    fig.savefig(args.out_file, bbox_inches="tight")