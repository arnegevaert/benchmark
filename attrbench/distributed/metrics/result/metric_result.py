from abc import abstractmethod
from typing import Tuple
from attrbench.distributed.metrics.result import BatchResult
import pandas as pd


class MetricResult:
    """
    Abstract class to represent results of distributed metrics.
    """
    def __init__(self, method_names: Tuple[str], shape: Tuple[int, ...]):
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

    def get_df(self, *args, **kwargs) -> Tuple[pd.DataFrame, bool]:
        """
        Retrieve a dataframe from the result object for some given arguments,
        along with a boolean indicating if higher is better.
        These arguments depend on the specific metric.
        """
        raise NotImplementedError
