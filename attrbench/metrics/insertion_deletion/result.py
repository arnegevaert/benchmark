from attrbench.metrics import MaskerActivationMetricResult
import numpy as np


class InsertionDeletionResult(MaskerActivationMetricResult):
    def _aggregate(self, data):
        return np.trapz(data, x=np.linspace(0, 1, data.shape[1]))


class InsertionResult(InsertionDeletionResult):
    inverted = False


class DeletionResult(InsertionDeletionResult):
    inverted = True


class IrofResult(MaskerActivationMetricResult):
    inverted = True


class IiofResult(MaskerActivationMetricResult):
    inverted = False
