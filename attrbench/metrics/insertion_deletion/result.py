from attrbench.metrics import MaskerActivationMetricResult
import numpy as np


class InsertionDeletionResult(MaskerActivationMetricResult):
    def _postproc_fn(self, x):
        return np.trapz(x, x=np.linspace(0, 1, x.shape[-1]))


class InsertionResult(InsertionDeletionResult):
    inverted = False


class DeletionResult(InsertionDeletionResult):
    inverted = True


class IrofResult(MaskerActivationMetricResult):
    inverted = True


class IiofResult(MaskerActivationMetricResult):
    inverted = False
