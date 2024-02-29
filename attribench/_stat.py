import numpy as np
from numpy import typing as npt
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pandas as pd
from typing import Tuple, Optional


def _wilcoxon_signed_rank_test(
    x: npt.NDArray,
    alternative: str,
):
    _, pvalue = stats.wilcoxon(x, alternative=alternative)
    return pvalue


def _t_test(
    x: npt.NDArray,
    alternative: str,
):
    _, pvalue = stats.ttest_1samp(x, popmean=0, alternative=alternative)
    return pvalue


def _sign_test(
    x: npt.NDArray,
    alternative: str,
):
    pos = np.sum(x >= 0)
    neg = np.sum(x <= 0)
    n = len(x)
    if alternative == "greater":
        pvalue = stats.binom_test(pos, n, p=0.5)
    elif alternative == "less":
        pvalue = stats.binom_test(neg, n, p=0.5)
    else:
        raise ValueError("alternative must be one of 'greater' or 'less'")
    return pvalue


def significance_tests(
    df: pd.DataFrame,
    alpha: float,
    test: str,
    alternative: str,
    multiple_testing: Optional[str] = None,
) -> Tuple[dict[str, float], dict[str, float]]:
    """
    Perform significance tests on a DataFrame of method results.
    The results are assumed to be differences to the baseline method.
    The function will perform a significance test for each method, comparing
    the differences to the baseline method to zero. The effect sizes will be
    calculated as the probability of superiority (i.e. the probability that a
    random observation from the method is greater than a random observation from
    the baseline method).

    Args:
        df (pd.DataFrame): DataFrame containing the method results.
        higher_is_better (bool): Flag indicating whether higher values are better.
        alpha (float): Significance level for the tests.
        test (str): Type of test to perform. Can be one of "wilcoxon", "paired_t_test", or "sign_test".
        alternative (str): The alternative hypothesis for the test. Can be one of "greater" or "less".
            Use "greater" if higher values are better, and "less" if lower values are better.
        multiple_testing (Optional[str], optional): Method for multiple testing correction.
            Can be one of "bonferroni", "fdr_bh", or None. Defaults to None.

    Returns:
        Tuple[dict[str, float], dict[str, float]]: A tuple containing two dictionaries.
            The first dictionary contains the effect sizes for each method.
            The second dictionary contains the p-values for each method.

    Raises:
        AssertionError: If `multiple_testing` or `test` is not one of the allowed values.
    """
    assert multiple_testing in ["bonferroni", "fdr_bh", None]
    assert test in ["wilcoxon", "t_test", "sign_test"]
    pvalues, effect_sizes = {}, {}
    for method_name in df:
        method_results = df[method_name].to_numpy()
        if test == "wilcoxon":
            pvalue = _wilcoxon_signed_rank_test(method_results, alternative)
        elif test == "t_test":
            pvalue = _t_test(method_results, alternative)
        elif test == "sign_test":
            pvalue = _sign_test(method_results, alternative)

        pvalues[method_name] = pvalue
        # Use probability of superiority as effect size
        effect_sizes[method_name] = np.mean(method_results > 0) if alternative == "greater" else np.mean(method_results < 0)

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
