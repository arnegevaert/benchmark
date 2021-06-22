import argparse
from attrbench.lib import krippendorff_alpha
from attrbench.suite import SuiteResult
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from os import path
import os


def generate_plot(_x_range, _metric_name):
    fig, ax = plt.subplots()

    for ds_name, res in datasets.items():
        k_alphas = []
        for i in _x_range:
            df, _ = res.metric_results[_metric_name].get_df(mode="single", columns=np.arange(i))
            k_alphas.append(krippendorff_alpha(df.to_numpy()))
        ax.plot(k_alphas, label=ds_name)
    fig.legend()
    fig.savefig(path.join(args.out_dir, f"{_metric_name}.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    mpl.use("Agg")
    np.seterr(all="raise")

    dataset_names = ["mnist", "fashionmnist", "cifar10", "cifar100", "svhn", "imagenet", "caltech", "places"]
    datasets = {ds_name: SuiteResult.load_hdf(path.join(args.in_dir, f"{ds_name}.h5")) for ds_name in dataset_names}
    step = 10

    if not path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    for metric_name in ["deletion_morf", "deletion_lerf", "irof_morf", "irof_lerf"]:
        print(metric_name)
        x_range = np.arange(3, 100, step=step)
        generate_plot(x_range, metric_name)

    for metric_name in ["sensitivity_n", "seg_sensitivity_n"]:
        print(metric_name)
        x_range = np.arange(1, 10)
        generate_plot(x_range, metric_name)
