from attrbench.metrics import ModeActivationMetricResult


class InsertionDeletionResult(ModeActivationMetricResult):
    pass


class InsertionResult(InsertionDeletionResult):
    pass


class DeletionResult(InsertionDeletionResult):
    pass


class IrofResult(InsertionDeletionResult):
    pass


class IiofResult(InsertionDeletionResult):
    pass
