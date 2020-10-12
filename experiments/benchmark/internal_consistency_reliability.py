import argparse
from experiments.benchmark import load_results, correlation_heatmap
import numpy as np
import matplotlib.pyplot as plt
import warnings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()

    all_data, _ = load_results(args.dir)

    """
    Internal consistency reliability: pairwise correlation between scores of a single method
    produced by different metrics on the images.
    """
    fig, axs = plt.subplots(4, 2, figsize=(15, 25))
    axs = axs.flatten()

    metrics = ["insertion", "deletion", "infidelity", "sens-n", "max-sens"]
    for i, method in enumerate(all_data):
        method_data = []
        # TODO max-sens could also be incorporated if we save per-sample results
        # TODO for insertion/deletion: should we normalize each sample to 0-1 or just take avg of logits?
        for metric in metrics:
            m_data = all_data[method][metric]
            if np.any(np.isnan(m_data)):
                warnings.warn(f"NaN found in {method}/{metric}")
            if metric == "deletion":
                method_data.append(np.mean(m_data[:, 0].reshape(-1, 1) - m_data, axis=1))
            elif metric == "insertion":
                method_data.append(np.mean(m_data - m_data[:, 0].reshape(-1, 1), axis=1))
            else:
                method_data.append(np.mean(m_data, axis=1))
        corrs = np.corrcoef(np.vstack(method_data))
        correlation_heatmap(axs[i], corrs, metrics, method)
    fig.tight_layout()
    fig.savefig(args.output)
