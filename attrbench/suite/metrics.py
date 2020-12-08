class Metric:
    def __init__(self, model, methods):
        self.model = model
        self.methods = methods

    def run_batch(self, samples, labels):
        """
        Runs the metric for a given batch, for all methods, and saves result internally
        """
        raise NotImplementedError


class DeletionUntilFlip(Metric):
    def __init__(self, model, methods):
        super().__init__(model, methods)

    def run_batch(self, samples, labels):
        pass


class ImpactCoverage(Metric):
    def __init__(self, model, methods):
        super().__init__(model, methods)

    def run_batch(self, samples, labels):
        pass


class ImpactScore(Metric):
    def __init__(self, model, methods):
        super().__init__(model, methods)

    def run_batch(self, samples, labels):
        pass


class Infidelity(Metric):
    def __init__(self, model, methods):
        super().__init__(model, methods)

    def run_batch(self, samples, labels):
        pass


class Insertion(Metric):
    def __init__(self, model, methods):
        super().__init__(model, methods)

    def run_batch(self, samples, labels):
        pass


class Deletion(Metric):
    def __init__(self, model, methods):
        super().__init__(model, methods)

    def run_batch(self, samples, labels):
        pass


class MaxSensitivity(Metric):
    def __init__(self, model, methods):
        super().__init__(model, methods)

    def run_batch(self, samples, labels):
        pass


class SensitivityN(Metric):
    def __init__(self, model, methods):
        super().__init__(model, methods)

    def run_batch(self, samples, labels):
        pass
