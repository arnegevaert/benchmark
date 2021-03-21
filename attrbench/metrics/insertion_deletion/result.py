from attrbench.metrics import ModeActivationMetricResult
import numpy as np


class InsertionDeletionResult(ModeActivationMetricResult):
    def _aggregate(self, data):
        return np.trapz(data, x=np.linspace(0, 1, data.shape[1]))


class DeletionResult(InsertionDeletionResult):
    inverted = {
        "morf": True,
        "lerf": False
    }


class IrofResult(ModeActivationMetricResult):
    inverted = {
        "morf": True,
        "lerf": False
    }


class InsertionResult(InsertionDeletionResult):
    inverted = {
        "morf": False,
        "lerf": True
    }


class IiofResult(ModeActivationMetricResult):
    inverted = {
        "morf": False,
        "lerf": True
    }
