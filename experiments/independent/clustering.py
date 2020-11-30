import argparse
from experiments.independent import Result
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    args = parser.parse_args()
    result = Result(args.dir)

    y = [np.mean(result.aggregate(method, "infidelity")) for method in result.method_names]
    y = np.log(y).reshape(-1, 1)
    print(y, y.shape)
    Z = hierarchy.linkage(y, "single")
    plt.figure()
    dn = hierarchy.dendrogram(Z, labels=result.method_names, orientation="left")
    plt.show()
