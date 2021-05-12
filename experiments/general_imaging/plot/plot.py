from attrbench.suite import SuiteResult
import numpy as np
from attrbench.suite.plot import *
from experiments.general_imaging.plot.dfs import get_default_dfs, get_metric_dfs
import matplotlib.pyplot as plt
import argparse
import os
from os import path
import matplotlib as mpl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("mode", type=str)
    parser.add_argument("-p", "--plot_types", type=str, nargs="*")
    parser.add_argument("--infid-log", action="store_true")
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all='raise')

    all_types = ["wsp", "krip", "krip_bs", "mad", "cluster", "metric_corr", "method_corr"]
    types = args.plot_types if args.plot_types is not None else all_types

    if not path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    res = SuiteResult.load_hdf(args.in_file)
    print(list(res.metric_results.keys()))

    metric_dfs = get_metric_dfs(res, mode="median_dist", infid_log=args.infid_log)
    default_dfs = get_default_dfs(res, mode="median_dist", infid_log=args.infid_log)

    # Wilcoxon summary plots
    if "wsp" in types:
        print("WSP...")
        if not path.isdir(path.join(args.out_dir, "wsp")):
            os.makedirs(path.join(args.out_dir, "wsp"))
        for metric_name, dfs in metric_dfs.items():
            fig = WilcoxonSummaryPlot(dfs).render(figsize=(20, 20), glyph_scale=1500)
            fig.savefig(path.join(args.out_dir, "wsp", f"{metric_name}.png"), bbox_inches="tight")
            plt.close(fig)

    # Krippendorff barplots
    if "krip" in types:
        print("Krippendorff...")
        if not path.isdir(path.join(args.out_dir, "krip")):
            os.makedirs(path.join(args.out_dir, "krip"))
        for metric_name, dfs in metric_dfs.items():
            fig = KrippendorffAlphaPlot(dfs).render()
            fig.savefig(path.join(args.out_dir, "krip", f"{metric_name}.png"), bbox_inches="tight")
            plt.close(fig)

    # Krippendorff bootstrap plots
    if "krip_bs" in types:
        print("Krippendorff Bootstrap...")
        if not path.isdir(path.join(args.out_dir, "krip_bs")):
            os.makedirs(path.join(args.out_dir, "krip_bs"))
        for metric_name, dfs in metric_dfs.items():
            fig = KrippendorffAlphaBootstrapPlot(dfs).render()
            fig.savefig(path.join(args.out_dir, "krip_bs", f"{metric_name}.png"), bbox_inches="tight")
            plt.close(fig)

    # MAD barplots
    if "mad" in types:
        print("MAD ratio...")
        if not path.isdir(path.join(args.out_dir, "mad")):
            os.makedirs(path.join(args.out_dir, "mad"))
        for metric_name, dfs in metric_dfs.items():
            fig = MADRatioPlot(dfs).render()
            fig.savefig(path.join(args.out_dir, "mad", f"{metric_name}.png"), bbox_inches="tight")
            plt.close(fig)

    # Clusterplots
    if "cluster" in types:
        print("Cluster...")
        fig = ClusterPlot(default_dfs).render()
        fig.savefig(path.join(args.out_dir, "cluster.png"), bbox_inches="tight")
        plt.close(fig)

    # Inter-metric correlation plots
    if "metric_corr" in types:
        print("Inter-metric correlation...")
        fig = InterMetricCorrelationPlot(default_dfs).render(glyph_scale=10000)
        fig.savefig(path.join(args.out_dir, "metric_corr.png"), bbox_inches="tight")
        plt.close(fig)

    # Inter-method correlation plots
    if "method_corr" in types:
        print("Inter-method correlation...")
        fig = InterMethodCorrelationPlot(default_dfs).render(glyph_scale=5000)
        fig.savefig(path.join(args.out_dir, "method_corr.png"), bbox_inches="tight")
        plt.close(fig)
