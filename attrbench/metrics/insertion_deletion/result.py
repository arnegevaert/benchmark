from attrbench.metrics import ActivationMetricResult
import numpy as np


class InsertionDeletionResult(ActivationMetricResult):
    def _aggregate(self, data):
        return np.trapz(data, x=np.linspace(0, 1, data.shape[1]))


class InsertionResult(InsertionDeletionResult):
    inverted = False


class DeletionResult(InsertionDeletionResult):
    inverted = True


class IrofResult(ActivationMetricResult):
    inverted = True


class IiofResult(ActivationMetricResult):
    inverted = False
