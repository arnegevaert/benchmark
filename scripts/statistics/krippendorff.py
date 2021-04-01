import argparse
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from attrbench.suite import SuiteResult
import logging
from scripts.statistics.util import krippendorff_alpha
from scripts.statistics.df_extractor import DFExtractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("out_file", type=str)
    args = parser.parse_args()

    # Constant parameters, might be moved to args if necessary
    EXCLUDE = ["Random_pos_only", "GradCAM_no_relu", "GuidedGradCAM_no_relu"]
    BASELINE = "Random"
    plt.rcParams["figure.dpi"] = 140
    RES_OBJ = SuiteResult.load_hdf(args.file)

    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S")

    dfe = DFExtractor(RES_OBJ, EXCLUDE)
    dfe.add_infidelity("mse", "linear")
    #dfe.add_infidelity("corr", "linear")
    dfe.compare_maskers(["constant", "blur", "random"], "linear")
    #dfe.compare_maskers(["constant"], "linear")
    dfs = dfe.get_dfs()

    k_a = {}
    for metric_name, (df, inverted) in dfs.items():
        df = df[df.columns.difference([BASELINE])]
        k_a[metric_name] = krippendorff_alpha(df.to_numpy())
    k_a = pd.DataFrame(k_a, index=["Krippendorff Alpha"]).transpose()
    fig, ax = plt.subplots()
    k_a.plot.barh(figsize=(10, 10), ax=ax)
    fig.tight_layout()
    fig.savefig(args.out_file)
