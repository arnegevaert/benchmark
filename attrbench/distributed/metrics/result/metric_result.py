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

    def get_df(self, methods: List[str] = None, *args) -> Tuple[pd.DataFrame, bool]:
        """
        Construct a pandas DataFrame for a given list of methods and possible other arguments (abstract method).
        :param methods: The methods to incude. If None, include all methods.
        :param args: Possible other arguments (depends on subclass).
        :return: Dataframe containing results and boolean indicating if higher is better.
        """
        raise NotImplementedError
