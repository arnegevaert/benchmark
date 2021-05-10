import numpy as np
from scipy import stats
from multiprocessing.pool import ThreadPool
import multiprocessing
from functools import partial
import numba as nb


@nb.njit()
def cohend(x, y, type="mean"):
    pooled_std = np.sqrt(((x.shape[0] - 1) * np.var(x) + (y.shape[0] - 1) * np.var(y)) / (x.shape[0] + y.shape[0] - 2))
    if type == "mean":
        return (np.mean(x) - np.mean(y)) / pooled_std
    elif type == "median":
        return (np.median(x) - np.median(y)) / pooled_std
    else:
        raise ValueError("Type must be mean or median")


def emp_power_curve(sample, baseline_sample, effect_size, iterations, n_range, inverted, tolerance, alpha):
    emp_power = []
    for n in n_range:
        bs_indices = np.random.choice(np.arange(sample.shape[0]), size=iterations * n, replace=True)
        bs_sample = sample[bs_indices].reshape(iterations, n)
        bs_baseline = baseline_sample[bs_indices].reshape(iterations, n)

        with ThreadPool(multiprocessing.cpu_count()) as p:
            results = p.starmap(partial(stats.wilcoxon, alternative="less" if inverted else "greater"),
                                [(bs_sample[i, ...], bs_baseline[i, ...]) for i in range(iterations)])
        pvalues = np.array([p[1] for p in results])
        effect_sizes = np.array([cohend(bs_sample[i, ...], bs_baseline[i, ...]) for i in range(iterations)])
        significant = pvalues < alpha
        effect = (effect_sizes <= tolerance * effect_size) \
            if inverted else (effect_sizes >= tolerance * effect_size)
        detected = significant & effect
        emp_power.append(detected.sum() / iterations)
    return emp_power


def wilcoxon_tests(df, baseline, effect_size_measure, inverted):
    pvalues, effect_sizes = {}, {}
    for method_name in df:
        method_results = df[method_name].to_numpy()
        statistic, pvalue = stats.wilcoxon(method_results, baseline,
                                           alternative="less" if inverted else "greater")
        pvalues[method_name] = pvalue
        if effect_size_measure == "cohend":
            effect_sizes[method_name] = cohend(method_results, baseline)
        elif effect_size_measure == "meandiff":
            effect_sizes[method_name] = np.mean(method_results) - np.mean(baseline)
        else:
            raise ValueError("Variant must be mse or corr")
    return effect_sizes, pvalues



