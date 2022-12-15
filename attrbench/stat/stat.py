import numpy as np
from scipy import stats
import warnings


def wilcoxon_tests(df, inverted):
    pvalues, effect_sizes = {}, {}
    for method_name in df:
        method_results = df[method_name].to_numpy()
        statistic, pvalue = stats.wilcoxon(method_results,
                                           alternative="less" if inverted else "greater")
        pvalues[method_name] = pvalue
        effect_sizes[method_name] = np.median(method_results)
    return effect_sizes, pvalues


def _interval_metric(a, b):
    return (a - b) ** 2


def krippendorff_alpha(data):
    # Assumptions: no missing values, interval metric, data is numpy array ([observers, samples])
    # Assuming no missing values, each column is a unit, and the number of pairable values is m*n
    pairable_values = data.shape[0] * data.shape[1]

    # Calculate observed disagreement
    observed_disagreement = 0.
    for col in range(data.shape[1]):
        unit = data[:, col].reshape(1, -1)
        observed_disagreement += np.sum(_interval_metric(unit, unit.T))
    observed_disagreement /= (pairable_values * (data.shape[0] - 1))

    # Calculate expected disagreement
    expected_disagreement = 0.
    for col1 in range(data.shape[1]):
        unit1 = data[:, col1].reshape(1, -1)
        for col2 in range(data.shape[1]):
            unit2 = data[:, col2].reshape(1, -1)
            expected_disagreement += np.sum(_interval_metric(unit1, unit2.T))
    expected_disagreement /= (pairable_values * (pairable_values - 1))
    return 1. - (observed_disagreement / expected_disagreement)


def corrcoef(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculates row-wise correlations between two arrays.
    :param a: first set of row vectors (shape: [num_rows, num_measurements])
    :param b: second set of row vectors (shape: [num_rows, num_measurements])
    :return: row-wise correlations between a and b (shape: [num_rows])
    """
    # Subtract mean
    # [batch_size, num_observations]
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    # Calculate numerator
    # [batch_size]
    cov = (a * b).sum(axis=1)
    # Calculate denominator
    # [batch_size]
    denom = np.sqrt((a**2).sum(axis=1)) * np.sqrt((b**2).sum(axis=1))
    denom_zero = (denom == 0.)
    if np.any(denom_zero):
        warnings.warn("Zero standard deviation detected.")
    corrcoefs = np.divide(cov, denom, out=np.zeros_like(cov), where=denom!=0)
    return corrcoefs

