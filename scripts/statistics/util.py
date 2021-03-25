import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def get_dfs(res_obj, metric_name, baseline_method, mode=None, activation=None, ignore_methods=None):
    df_dict = res_obj.metric_results[metric_name].to_df()
    df = df_dict[f"{mode}_{activation}"]
    inverted = res_obj.metric_results[metric_name].inverted[mode]
    if ignore_methods is not None:
        df = df[df.columns.difference(ignore_methods)]
    baseline = df[baseline_method]
    df = df[df.columns.difference([baseline_method])]
    return df, baseline, inverted


def cohend(x, y, type="mean"):
    pooled_std = np.sqrt(((x.shape[0] - 1) * np.var(x) + (y.shape[0] - 1) * np.var(y)) / (x.shape[0] + y.shape[0] - 2))
    if type == "mean":
        return (np.mean(x) - np.mean(y)) / pooled_std
    elif type == "median":
        return (np.median(x) - np.median(y)) / pooled_std
    else:
        raise ValueError("Type must be mean or median")


def plot_wilcoxon_result(title, effect_sizes, pvalues, labels):
    fig, axs = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [4, 1]})

    effect_sizes.plot.barh(figsize=(14, 6), title=f"{title} effect sizes", ax=axs[0])
    axs[0].legend(labels, loc='upper center', bbox_to_anchor=(0.5, -.05), ncol=3, fancybox=True,
                  shadow=True)
    axs[1].pcolor(pvalues < 0.05, cmap="RdYlGn", edgecolor="black")
    axs[1].set_title("p < 0.05")
    axs[1].set_yticks([])
    axs[1].set_xticks(np.arange(3) + 0.5)
    axs[1].tick_params(axis="x", rotation=45)
    axs[1].set_xticklabels(labels, ha="right")
    plt.show()


def emp_power_curve(sample, baseline_sample, effect_size, iterations, n_range, inverted, tolerance, alpha):
    emp_power = []
    for n in n_range:
        bs_indices = np.random.choice(np.arange(sample.shape[0]), size=(iterations, n))
        bs_sample = sample[bs_indices]
        bs_baseline = baseline_sample[bs_indices]
        pvalues = [stats.wilcoxon(bs_sample[i, ...], bs_baseline[i, ...], alternative="less" if inverted else "greater")[1]
                   for i in range(iterations)]
        effect_sizes = [cohend(bs_sample[i, ...], bs_baseline[i, ...]) for i in range(iterations)]
        significant = np.array(pvalues) < alpha
        effect = (np.array(effect_sizes) <= tolerance * effect_size) \
            if inverted else (np.array(effect_sizes) >= tolerance * effect_size)
        detected = significant & effect
        emp_power.append(detected.sum() / iterations)
    return emp_power
