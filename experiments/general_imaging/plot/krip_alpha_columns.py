import argparse
from attrbench.lib import krippendorff_alpha
from attrbench.suite import SuiteResult
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from os import path
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    #mpl.use("Agg")
    #np.seterr(all="raise")

    if not path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    res = SuiteResult.load_hdf(args.in_file)
    k_alphas = []
    for i in range(1, 100):
        df, _ = res.metric_results["irof_lerf"].get_df(mode="single", columns=np.arange(i))
        k_alphas.append(krippendorff_alpha(df.to_numpy()))
    plt.plot(k_alphas)
    plt.show()