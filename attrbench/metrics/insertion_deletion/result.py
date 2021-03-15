from attrbench.metrics import ModeActivationMetricResult


class InsertionDeletionResult(ModeActivationMetricResult):
    pass


class InsertionResult(InsertionDeletionResult):
    inverted = {
        "morf": False,
        "lerf": True
    }


class DeletionResult(InsertionDeletionResult):
    inverted = {
        "morf": True,
        "lerf": False
    }


class IrofResult(InsertionDeletionResult):
    inverted = {
        "morf": True,
        "lerf": False
    }


class IiofResult(InsertionDeletionResult):
    inverted = {
        "morf": False,
        "lerf": True
    }
