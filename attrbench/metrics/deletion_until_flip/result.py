from attrbench.metrics import MetricResult
from typing import List


class DeletionUntilFlipResult(MetricResult):
    def __init__(self, method_names: List[str]):
        super().__init__(method_names)
        self.inverted = True
