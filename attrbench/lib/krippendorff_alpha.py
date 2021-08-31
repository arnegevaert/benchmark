import numpy as np
import krippendorff
from scipy.stats import rankdata


def _interval_metric(a, b):
    return (a - b) ** 2


def krippendorff_alpha(data):
    return krippendorff.alpha(rankdata(data, axis=1), level_of_measurement="ordinal")
