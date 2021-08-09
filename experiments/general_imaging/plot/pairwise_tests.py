import argparse
from attrbench.suite import SuiteResult
from experiments.general_imaging.plot.dfs import get_all_dfs
from pingouin import wilcoxon


# TODO find a good way to plot the result of a large number of pairwise wilcoxon tests with common language effect sizes
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf_file", type=str)
    parser.add_argument("mth1", type=str)
    parser.add_argument("mth2", type=str)
    args = parser.parse_args()

    res_obj = SuiteResult.load_hdf(args.hdf_file)
    dfs = get_all_dfs(res_obj, "single")

    for key in dfs:
        df, inverted = dfs[key]

        res = wilcoxon(x=df[args.mth1], y=df[args.mth2], tail="one-sided")
        pvalue = res["p-val"]["Wilcoxon"]
        cles = res["CLES"]["Wilcoxon"]
        out_str = f"{key}: p={pvalue:.3f} CLES={cles:.3f}"
        out_str += " *" if pvalue < 0.01 else " -"
        print(out_str)
