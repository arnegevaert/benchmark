import numpy as np
from numpy import typing as npt
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pandas as pd
from typing import Tuple, Optional


def wilcoxon_tests(
    df: pd.DataFrame,
    higher_is_better: bool,
    alpha: float,
    multiple_testing: Optional[str] = None,
):
    """Perform Wilcoxon signed-rank tests on a dataframe of results.
    The dataframe contains results for each method on a given metric.
    Contents of the dataframe are interpreted as differences between
    the method's performance and the random baseline's performance on
    the metric (i.e. the random baseline must be subtracted from the
    method's performance before passing the results to this function).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of results. Each column is a method and each row is a sample.
    higher_is_better : bool
        Whether higher values are better for the metric.
    alpha : float
        Significance level.
    multiple_testing : str
        Multiple testing correction method to use.
        One of "bonferroni" or "fdr_bh" (Benjamini-Hochberg).

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float]]
        Tuple of dictionaries mapping method names to effect sizes and p-values.
    """
    assert multiple_testing in ["bonferroni", "fdr_bh", None]
    pvalues, effect_sizes = {}, {}
    for method_name in df:
        method_results = df[method_name].to_numpy()
        _, pvalue = stats.wilcoxon(
            method_results,
            alternative="greater" if higher_is_better else "less",
        )
        pvalues[method_name] = pvalue
        effect_sizes[method_name] = np.median(method_results)

    if multiple_testing is not None:
        # Build list of method names and p-values in the same order
        # to make sure they are matched up correctly after correction using
        # statsmodels.stats.multitest.multipletests
        method_names = []
        pvalues_list = []
        for key in pvalues:
            method_names.append(key)
            pvalues_list.append(pvalues[key])

        _, pvals_corrected, _, _ = multipletests(
            pvalues_list,
            method=multiple_testing,
            alpha=alpha,
        )

        for i, method_name in enumerate(method_names):
            pvalues[method_name] = pvals_corrected[i]

    return effect_sizes, pvalues


def rowwise_pearsonr(
    a: npt.NDArray,
    b: npt.NDArray,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Calculates row-wise correlations between two arrays.
    This is a faster implementation of scipy.stats.pearsonr,
    as it only calculates correlation coefficients between corresponding rows,
    rather than between all pairs of rows.

    Parameters
    ----------
    a : npt.NDArray
        first set of row vectors (shape: [num_rows, num_measurements])
    b : npt.NDArray
        second set of row vectors (shape: [num_rows, num_measurements])

    Returns
    -------
    npt.NDArray
        row-wise Pearson correlations between a and b (shape: [num_rows])
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
    # denom_zero = denom == 0.0
    # if np.any(denom_zero):
    #    warnings.warn(f"Zero standard deviation detected")

    # If the denominator is zero, that means one of the series is constant.
    # Correlation is technically undefined in this case, but covariance is 0.
    # We can just set the correlation coefficient to zero in this case.
    corrcoefs = np.divide(cov, denom, out=np.zeros_like(cov), where=denom != 0)

    return corrcoefs


def rowwise_spearmanr(
    a: npt.NDArray,
    b: npt.NDArray,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Calculates row-wise Spearman correlations between two arrays.
    This is a faster implementation of scipy.stats.spearmanr,
    as it only calculates correlation coefficients between corresponding rows,
    rather than between all pairs of rows.

    Parameters
    ----------
    a : npt.NDArray
        first set of row vectors (shape: [num_rows, num_measurements])
    b : npt.NDArray
        second set of row vectors (shape: [num_rows, num_measurements])

    Returns
    -------
    npt.NDArray
        row-wise Spearman correlations between a and b (shape: [num_rows])
    """
    # Calculate rank of each row
    # [batch_size, num_observations]
    a_ranks = stats.rankdata(a, axis=1)
    b_ranks = stats.rankdata(b, axis=1)

    # Spearman rank correlation is simply Pearson correlation of ranks
    return rowwise_pearsonr(a_ranks, b_ranks)
