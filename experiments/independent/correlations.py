import argparse
import numpy as np
import warnings
import matplotlib.pyplot as plt
import os
from os import path
import pandas as pd
import ppscore


def correlation_heatmap(ax, corrs, names=None, title=None):
    if title:
        ax.set_title(title)
    ax.imshow(corrs, vmin=-1, vmax=1)

    if names:
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(names)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{corrs[i, j]:.3f}",
                        ha="center", va="center", color="w")


def normalize(m_data):
    if np.any(np.isnan(m_data)):
        warnings.warn(f"NaN found in {ds}")
    if metric == "deletion":
        return np.mean(m_data[:, 0].reshape(-1, 1) - m_data, axis=1)
    elif metric == "insertion":
        return np.mean(m_data - m_data[:, 0].reshape(-1, 1), axis=1)
    else:
        return np.mean(m_data, axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    datasets = os.listdir(args.data_dir)
    metrics = ["insertion", "deletion", "infidelity", "sens-n", "max-sens"]

    for ds in datasets:
        print(f"{ds}...")
        all_data, _ = load_results(path.join(args.data_dir, ds))

        """
        Inter-method reliability: pairwise correlation between scores of different methods
        produced by a single metric on the images.
        """
        print(f"\tIMR...")
        fig, axs = plt.subplots(3, 2, figsize=(20, 20))
        axs = axs.flatten()

        for i, metric in enumerate(metrics):
            raw_metric_data = get_metric(all_data, metric)
            methods = list(raw_metric_data.keys())
            metric_data = [normalize(raw_metric_data[method]) for method in methods]
            corrs = np.corrcoef(np.vstack(metric_data))
            correlation_heatmap(axs[i], corrs, methods, metric)
        fig.tight_layout()
        plt.savefig(path.join(args.out_dir, f"{ds.lower()}-imr.png"))

        """
        Internal consistency reliability: pairwise correlation between scores of a single method
        produced by different metrics on the images.
        """
        print(f"\tICR...")
        fig, axs = plt.subplots(5, 2, figsize=(15, 25))
        axs = axs.flatten()

        for i, method in enumerate(all_data):
            method_data = [normalize(all_data[method][metric]) for metric in metrics]
            # TODO for insertion/deletion: should we normalize each sample to 0-1 or just take avg of logits?
            corrs = np.corrcoef(np.vstack(method_data))
            correlation_heatmap(axs[i], corrs, metrics, method)
        fig.tight_layout()
        fig.savefig(path.join(args.out_dir, f"{ds.lower()}-icr.png"))

        """
        Predictive Power score between metrics for each method
        """
        print(f"\tPPS...")
        fig, axs = plt.subplots(5, 2, figsize=(15, 25))
        axs = axs.flatten()

        for i, method in enumerate(all_data):
            df = pd.DataFrame()
            for metric in metrics:
                df[metric] = normalize(all_data[method][metric])
            matrix = ppscore.matrix(df)
            corrs = matrix[["x", "y", "ppscore"]].pivot(columns="x", index="y", values="ppscore")
            correlation_heatmap(axs[i], corrs.to_numpy(), list(corrs.columns), method)
        fig.tight_layout()
        fig.savefig(path.join(args.out_dir, f"{ds.lower()}-pps.png"))
