from attrbench.suite import SuiteResult
from tqdm import tqdm
from os import path
import os
from scripts.statistics.df_extractor import DFExtractor
from scripts.statistics import correlation, clustering, wilcoxon, krippendorff, boxplot
import matplotlib as mpl
from matplotlib import pyplot as plt
import argparse


def get_dfs():
    pass


if __name__ == "__main__":
    mpl.use("agg")
    plot_choices = ["correlation", "wilcoxon", "wilcoxon-summary", "boxplot", "clustering", "krippendorff", "krippendorff-sig"]
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

    dfe = DFExtractor(RES_OBJ, EXCLUDE)
    if "impact_coverage" in RES_OBJ.metric_results.keys():
        dfe.add_metric("impact_coverage", "impact_coverage")
    dfe.add_infidelity("mse", "linear")
    dfe.compare_maskers(["constant", "blur", "random"], "linear")
    for plot in tqdm(args.plots):
        if plot == "wilcoxon":
            metric_groups = ["deletion_until_flip", "insertion", "deletion", "irof", "iiof", "seg_sensitivity_n", "sensitivity_n"]
            for m_group in metric_groups:
                dfe_wilc = DFExtractor(RES_OBJ, EXCLUDE)
                dfe_wilc.compare_maskers(["constant", "blur", "random"], "linear", metric_group=m_group)
                es_measure = "meandiff" if "sensitivity_n" in m_group else "cohend"
                wilcoxon(dfe_wilc, es_measure, BASELINE, path.join(args.out_dir, f"{m_group}_wilcoxon.png"))

            dfe_wilc = DFExtractor(RES_OBJ, EXCLUDE)
            dfe_wilc.add_infidelity("mse", "linear")
            wilcoxon(dfe_wilc, "cohend", BASELINE, path.join(args.out_dir, "infidelity_wilcoxon.png"))

            if "impact_coverage" in RES_OBJ.metric_results.keys():
                dfe_wilc = DFExtractor(RES_OBJ, EXCLUDE)
                dfe_wilc.add_metric("impact_coverage", "impact_coverage")
                wilcoxon(dfe_wilc, "meandiff", BASELINE, path.join(args.out_dir, "impact_coverage_wilcoxon.png"))

        if plot == "boxplot":
            metric_groups = ["deletion_until_flip", "insertion", "deletion", "irof", "iiof", "seg_sensitivity_n", "sensitivity_n"]
            for m_group in metric_groups:
                dfe_box = DFExtractor(RES_OBJ, EXCLUDE)
                dfe_box.compare_maskers(["constant", "blur", "random"], "linear", metric_group=m_group)
                boxplot(dfe_box, path.join(args.out_dir, f"{m_group}_box.png"))

            dfe_box = DFExtractor(RES_OBJ, EXCLUDE)
            dfe_box.add_infidelity("mse", "linear")
            boxplot(dfe_box, path.join(args.out_dir, f"infidelity_box.png"))

            if "impact_coverage" in RES_OBJ.metric_results.keys():
                dfe_box = DFExtractor(RES_OBJ, EXCLUDE)
                dfe_box.add_metric("impact_coverage", "impact_coverage")
                boxplot(dfe_box, path.join(args.out_dir, f"impact_coverage_box.png"))

        if plot == "clustering":
            clustering(dfe, args.out_dir, BASELINE)
        if plot == "krippendorff":
            krippendorff(dfe, BASELINE, path.join(args.out_dir, "krippendorff.png"))
        if plot == "krippendorff-sig":
            krippendorff(dfe, BASELINE, path.join(args.out_dir, "krippendorff_sig.png"), exclude_non_significant=True)
        if plot == "wilcoxon-summary":
            pass
        if plot == "correlation":
            correlation(dfe, args.out_dir, BASELINE)
