from attrbench.metrics import MaskerActivationMetricResult
import numpy as np


class DeletionResult(MaskerActivationMetricResult):
    inverted = True

    def _postproc_fn(self, x):
        return np.trapz(x, x=np.linspace(0, 1, x.shape[-1]))


class IrofResult(MaskerActivationMetricResult):
    inverted = True
