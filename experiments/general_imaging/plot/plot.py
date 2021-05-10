from attrbench.suite import SuiteResult
from attrbench.suite.plot import *
import matplotlib.pyplot as plt
from itertools import product
import argparse
import os
from os import path
import matplotlib as mpl


def krippendorff_plots(out_dir):
    if not path.isdir(out_dir):
        os.makedirs(out_dir)

    for mode in ("single_dist", "median_dist", "std_dist"):
        print(mode)
        for m_name in ("deletion", "insertion", "irof", "iiof", "sensitivity_n", "seg_sensitivity_n"):
            m_res = res.metric_results[m_name]
            dfs = {
                f"{afn} - {masker}": m_res.get_df(mode=mode, activation_fn=afn, masker=masker)
                for afn, masker in product(("linear", "softmax"), ("constant", "random", "blur"))
            }
            wsp = KrippendorffAlphaBootstrapPlot(dfs)
            fig = wsp.render()
            fig.savefig(path.join(out_dir, f"{m_name}-{mode}.png"), bbox_inches="tight")
            plt.close(fig)

        if mode != "std_dist":
            del_flip_res = res.metric_results["deletion_until_flip"]
            dfs = {
                f"{masker}": del_flip_res.get_df(mode=mode, masker=masker)
                for masker in ("constant", "random", "blur")
            }
            wsp = KrippendorffAlphaBootstrapPlot(dfs)
            fig = wsp.render()
            fig.savefig(path.join(out_dir, f"deletion_until_flip-{mode}.png"), bbox_inches="tight")
            plt.close(fig)

        infid_res = res.metric_results["infidelity"]
        dfs = {
            f"{pert_gen} - {afn} - {loss_fn}": infid_res.get_df(mode=mode, perturbation_generator=pert_gen,
                                                                activation_fn=afn, loss_fn=loss_fn)
            for (pert_gen, afn, loss_fn) in product(("gaussian", "square", "segment"),
                                                    ("linear", "softmax"),
                                                    ("mse", "normalized_mse", "corr"))
        }
        wsp = KrippendorffAlphaBootstrapPlot(dfs)
        fig = wsp.render()
        fig.savefig(path.join(out_dir, f"infidelity-{mode}.png"), bbox_inches="tight")


def wilcoxon_summary_plots(out_dir):
    if not path.isdir(out_dir):
        os.makedirs(out_dir)

    for mode in ("single_dist", "median_dist", "std_dist"):
        print(mode)
        for m_name in ("deletion", "insertion", "irof", "iiof", "sensitivity_n", "seg_sensitivity_n"):
            m_res = res.metric_results[m_name]
            dfs = {
                f"{afn} - {masker}": m_res.get_df(mode=mode, activation_fn=afn, masker=masker)
                for afn, masker in product(("linear", "softmax"), ("constant", "random", "blur"))
            }
            wsp = WilcoxonSummaryPlot(dfs)
            fig = wsp.render(figsize=(20, 15), glyph_scale=2000)
            fig.savefig(path.join(out_dir, f"{m_name}-{mode}.png"), bbox_inches="tight")
            plt.close(fig)

        if mode != "std_dist":
            del_flip_res = res.metric_results["deletion_until_flip"]
            dfs = {
                f"{masker}": del_flip_res.get_df(mode=mode, masker=masker)
                for masker in ("constant", "random", "blur")
            }
            wsp = WilcoxonSummaryPlot(dfs)
            fig = wsp.render(figsize=(20, 15), glyph_scale=2000)
            fig.savefig(path.join(out_dir, f"deletion_until_flip-{mode}.png"), bbox_inches="tight")
            plt.close(fig)
        infid_res = res.metric_results["infidelity"]
        dfs = {
            f"{pert_gen} - {afn} - {loss_fn}": infid_res.get_df(mode=mode, perturbation_generator=pert_gen,
                                                                activation_fn=afn, loss_fn=loss_fn)
            for (pert_gen, afn, loss_fn) in product(("gaussian", "square", "segment"),
                                                    ("linear", "softmax"),
                                                    ("mse", "normalized_mse", "corr"))
        }
        wsp = WilcoxonSummaryPlot(dfs)
        fig = wsp.render(figsize=(20, 20), glyph_scale=1500)
        fig.savefig(path.join(out_dir, f"infidelity-{mode}.png"), bbox_inches="tight")


def mad_ratio(df):
    # MAD = median of absolute deviation to median

    # Median of MADs for each method
    within_mad = df.sub(df.median()).abs().median().median()

    # MAD of all samples to global median
    global_median = df.stack().median()
    between_mad = df.stack().sub(global_median).abs().median()

    return between_mad / within_mad


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", type=str, nargs="?", default="../../../out/fashionmnist.h5")
    parser.add_argument("out_dir", type=str, nargs="?", default="out")
    args = parser.parse_args()

    mpl.use("Agg")

    if not path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    res = SuiteResult.load_hdf(args.in_file)
    print(list(res.metric_results.keys()))

    #wilcoxon_summary_plots(path.join(args.out_dir, "wsp"))
    krippendorff_plots(path.join(args.out_dir, "alpha"))
