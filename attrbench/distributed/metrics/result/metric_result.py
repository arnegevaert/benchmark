from typing import List, Tuple
from batch_result import BatchResult
import pandas as pd


class MetricResult:
    """
    Abstract class to represent results of distributed metrics.
    """
    def __init__(self, method_names: List[str], shape: Tuple[int, ...]):
        self.shape = shape
        self.method_names = method_names

    def add(self, batch_result: BatchResult):
        """
        Add a batch to the result (abstract method).
        """
        raise NotImplementedError

    def save(self, path: str):
        """
        Save the result to an HDF5 file (abstract method).
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "MetricResult":
        """
        Load a result from an HDF5 file (abstract method).
        """
        raise NotImplementedError
