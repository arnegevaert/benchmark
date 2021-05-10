import numpy as np


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
