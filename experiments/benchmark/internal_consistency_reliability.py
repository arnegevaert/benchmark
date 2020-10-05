import argparse
from experiments.benchmark import load_results
import numpy as np
import matplotlib.pyplot as plt
import warnings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()

    all_data = load_results(args.dir)

    """
    Internal consistency reliability: pairwise correlation between scores of a single method
    produced by different metrics on the images.
    """
    metrics = ["insertion", "deletion", "infidelity", "sens-n"]
    all_corrs = {}
    for method in all_data:
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
        all_corrs[method] = corrs

    fig, axs = plt.subplots(4, 2, figsize=(15, 25))
    axs = axs.flatten()
    for i, method in enumerate(all_corrs):
        ax = axs[i]
        ax.set_title(method)
        im = ax.imshow(all_corrs[method])

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(metrics)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(metrics)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f"{all_corrs[method][i, j]:.3f}",
                               ha="center", va="center", color="w")

    fig.tight_layout()
    fig.savefig(args.output)
    #plt.show()
