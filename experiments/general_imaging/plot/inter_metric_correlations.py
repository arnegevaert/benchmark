import argparse
from attrbench.suite import SuiteResult
from attrbench.suite.plot import InterMetricCorrelationPlot
from experiments.general_imaging.plot.dfs import get_default_dfs, get_all_dfs
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
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")

    if not path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    result_files = glob.glob(path.join(args.in_dir, "*.h5"))
    prog = tqdm(result_files)
    for file in prog:
        basename = path.basename(file)
        ds_name, ext = basename.split(".")
        prog.set_postfix_str(ds_name)
        mode = "raw"
        res_obj = SuiteResult.load_hdf(file)
        if args.all:
            dfs = get_all_dfs(res_obj, mode=mode)
        else:
            dfs = get_default_dfs(res_obj, mode=mode)

        figsize = (35, 35) if args.all else (10, 10)
        fig = InterMetricCorrelationPlot(dfs).render(figsize=figsize)
        ax = fig.axes[0]
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fig.savefig(path.join(args.out_dir, f"{ds_name}.png"), bbox_inches="tight")
        plt.close(fig)
