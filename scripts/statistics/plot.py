# Sources:
# https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
# https://seaborn.pydata.org/generated/seaborn.heatmap.html
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def plot_wilcoxon_result(effect_sizes, pvalues, labels, alpha):
    fig, axs = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [4, 1]})

    effect_sizes.plot.barh(figsize=(14, 6), ax=axs[0])
    axs[0].legend(labels, loc='upper center', bbox_to_anchor=(0.5, -.05), ncol=3, fancybox=True,
                  shadow=True)
    axs[1].pcolor(pvalues < alpha, cmap="RdYlGn", edgecolor="black",vmin=0., vmax=1.)
    axs[1].set_title(f"p < {alpha}")
    axs[1].set_yticks([])
    axs[1].set_xticks(np.arange(len(labels)) + 0.5)
    axs[1].tick_params(axis="x", rotation=45)
    axs[1].set_xticklabels(labels, ha="right")
    return fig, axs


def corr_heatmap(df):
    corr = df.corr(method=lambda x, y: pearsonr(x, y)[0])
    pvalues = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(corr.shape[0])
    corr[pvalues > 0.01] = 0

    corr = pd.melt(corr.reset_index(),
                   id_vars='index')  # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']
    fig = heatmap(
        x=corr['x'],
        y=corr['y'],
        size=corr['value'].abs(),
        color=corr['value']
    )
    return fig


def heatmap(x, y, size, color):
    sns.set()
    fig = plt.figure(figsize=(10, 10))
    fig.tight_layout()
    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1, figure=fig)  # 1x15 grid
    ax = fig.add_subplot(plot_grid[:, :-1])  # Use leftmost 14 columns for the main plot

    # Mapping from colnames to integer coordinates
    x_labels = list(sorted(x.unique()))
    y_labels = list(sorted(y.unique()))[::-1]
    x_to_num = {label: num for num, label in enumerate(x_labels)}
    y_to_num = {label: num for num, label in enumerate(y_labels)}

    n_colors = 256
    palette = sns.diverging_palette(230, 20, n=n_colors)
    color_min, color_max = -1, 1

    def value_to_color(val):
        val_position = float((val - color_min)) / (color_max - color_min)
        ind = int(val_position * (n_colors - 1))
        return palette[ind]

    size_scale = 10000 / max(len(x_labels), len(y_labels))
    ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        s=size * size_scale,  # Vector of square sizes, proportional to size parameter
        c=color.apply(value_to_color),
        marker='o'  # Use square as scatterplot marker
    )
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.set_xlim([-0.5, max(x_to_num.values()) + 0.5])
    ax.set_ylim([-0.5, max(y_to_num.values()) + 0.5])

    # Add color legend on the right side of the plot
    legend_ax = fig.add_subplot(plot_grid[:, -1])  # Use the rightmost column of the plot

    col_x = [0] * len(palette)  # Fixed x coordinate for the bars
    bar_y = np.linspace(color_min, color_max, n_colors)  # y coordinates for each of the n_colors bars

    bar_height = bar_y[1] - bar_y[0]
    legend_ax.barh(
        y=bar_y,
        width=[5] * len(palette),  # Make bars 5 units wide
        left=col_x,  # Make bars start at 0
        height=bar_height,
        color=palette,
        linewidth=0
    )
    legend_ax.set_xlim(1, 2)  # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
    legend_ax.grid(False)  # Hide grid
    legend_ax.set_facecolor('white')  # Make background white
    legend_ax.set_xticks([])  # Remove horizontal ticks
    legend_ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))  # Show vertical ticks for min, middle and max
    legend_ax.yaxis.tick_right()  # Show vertical ticks on the right
    return fig
