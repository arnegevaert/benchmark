from attrbench.suite import SuiteResult
from os import path
import os
import logging
from scripts.statistics.df_extractor import DFExtractor
from scripts.statistics import correlation, clustering
import matplotlib as mpl
from matplotlib import pyplot as plt
import argparse


if __name__ == "__main__":
    mpl.use("agg")
    plot_choices = ["correlation", "wilcoxon", "boxplot", "clustering", "krippendorff"]
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--plots", nargs="*", type=str, choices=plot_choices)
    args = parser.parse_args()
    if args.plots is None:
        args.plots = plot_choices

    # Constant parameters, might be moved to args if necessary
    EXCLUDE = ["Random_pos_only", "GradCAM_no_relu", "GuidedGradCAM_no_relu"]
    BASELINE = "Random"
    plt.rcParams["figure.dpi"] = 140
    RES_OBJ = SuiteResult.load_hdf(args.file)

    if not path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S")

    dfe = DFExtractor(RES_OBJ, EXCLUDE)
    dfe.add_infidelity("mse", "linear")
    dfe.compare_maskers(["constant", "blur", "random"], "linear")
    for plot in args.plots:
        print(plot)
        if plot == "correlation":
            correlation(dfe, args.out_dir, BASELINE)
        if plot == "wilcoxon":
            pass
        if plot == "boxplot":
            pass
        if plot == "clustering":
            clustering(dfe, args.out_dir, BASELINE)
        if plot == "krippendorff":
            pass
