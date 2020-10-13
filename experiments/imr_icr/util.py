import numpy as np
import matplotlib.pyplot as plt


def correlation_heatmap(ax, corrs, names, title):
    ax.set_title(title)
    ax.imshow(corrs)

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
