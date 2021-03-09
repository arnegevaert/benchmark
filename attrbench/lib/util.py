import numpy as np
import warnings


def corrcoef(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculates row-wise correlations between two arrays.
    :param a: first set of row vectors (shape: [num_rows, num_measurements])
    :param b: second set of row vectors (shape: [num_rows, num_measurements])
    :return: row-wise correlations between a and b (shape: [num_rows])
    """
    # Subtract mean
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    # Calculate covariances
    cov = (a * b).sum(axis=1) / (a.shape[1] - 1)
    # Divide by product of standard deviations
    # [batch_size]
    denom = a.std(axis=1) * b.std(axis=1)
    denom_zero = (denom == 0.)
    if np.any(denom_zero):
        warnings.warn("Zero standard deviation detected.")
    corrcoefs = cov / (a.std(axis=1) * b.std(axis=1))
    corrcoefs[denom_zero] = 0.
    return corrcoefs
