import seaborn as sns
from experiments.independent import Result
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind
import argparse
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    args = parser.parse_args()
    result = Result(args.dir)

    metrics = [m for m in result.metric_names if m not in ("impact", "s-impact")]
    methods = result.method_names

    metric = "infidelity"
    d1 = result.aggregate("Gradient", metric)
    d2 = result.aggregate("IntegratedGradients", metric)
    diff = d2 - d1

    plt.hist(d1, bins=100, density=True, alpha=0.6, color="b")
    plt.hist(d2, bins=100, density=True, alpha=0.6, color="r")
    plt.show()
    plt.hist(diff, bins=100, density=True, alpha=0.6, color="b")
    plt.show()

    print(ttest_rel(d1, d2))
    print(ttest_ind(d1, d2))
