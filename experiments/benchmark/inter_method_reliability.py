import argparse
from experiments.benchmark import load_results, get_metric, correlation_heatmap
import numpy as np
import warnings
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()

    all_data, _ = load_results(args.dir)

    """
    Inter-method reliability: pairwise correlation between scores of different methods
    produced by a single metric on the images.
    """
    fig, axs = plt.subplots(3, 2, figsize=(20, 20))
    axs = axs.flatten()

    metrics = ["insertion", "deletion", "infidelity", "sens-n", "max-sens"]
    for i, metric in enumerate(metrics):
        raw_metric_data = get_metric(all_data, metric)
        metric_data = []
        methods = list(raw_metric_data.keys())
        for method in methods:
            m_data = raw_metric_data[method]
            if np.any(np.isnan(m_data)):
                warnings.warn(f"NaN found in {method}/{metric}")
            if metric == "deletion":
                metric_data.append(np.mean(m_data[:, 0].reshape(-1, 1) - m_data, axis=1))
            elif metric == "insertion":
                metric_data.append(np.mean(m_data - m_data[:, 0].reshape(-1, 1), axis=1))
            else:
                metric_data.append(np.mean(m_data, axis=1))
        corrs = np.corrcoef(np.vstack(metric_data))
        correlation_heatmap(axs[i], corrs, methods, metric)
    fig.tight_layout()
    plt.savefig(args.output)