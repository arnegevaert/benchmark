from attrbench.metrics import ModeActivationMetricResult


class InsertionDeletionResult(ModeActivationMetricResult):
    pass


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


class InsertionResult(InsertionDeletionResult):
    inverted = {
        "morf": False,
        "lerf": True
    }


class IiofResult(InsertionDeletionResult):
    inverted = {
        "morf": False,
        "lerf": True
    }
