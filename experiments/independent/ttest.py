import seaborn as sns
from experiments.independent import Result
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind, norm
import argparse
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    args = parser.parse_args()
    result = Result(args.dir)

    metrics = [m for m in result.metric_names if m not in ("impact", "s-impact")]
    methods = result.method_names

    metric = "deletion"
    #d1 = result.aggregate("Random", metric)
    print(result.metadata)
    d1 = result.get_metric(metric)["Random"][:, 4]
    d2 = result.aggregate("IntegratedGradients", metric)
    diff = d2 - d1

    plt.hist(d1, bins=25, density=True, alpha=0.6, color="b")
    mu, std = norm.fit(d1)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    #plt.hist(d2, bins=100, density=True, alpha=0.6, color="r")
    plt.show()
    plt.hist(diff, bins=100, density=True, alpha=0.6, color="b")
    plt.show()

    print(ttest_rel(d1, d2))
    print(ttest_ind(d1, d2))
