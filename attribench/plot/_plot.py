from abc import abstractmethod
from typing import Tuple, Dict
import pandas as pd
from matplotlib.figure import Figure


class Plot:
    """Abstract base class for all plots.
    A plot is simply an object that has a render method that returns a 
    matplotlib Figure.
    """
    def __init__(self, dfs: Dict[str, Tuple[pd.DataFrame, bool]]):
        """
        Parameters
        ----------
        dfs : Dict[str, Tuple[pd.DataFrame, bool]]
            A dictionary mapping metric names to tuples of dataframes and
            booleans. The boolean indicates whether higher values of the metric
            are better (``True``) or not (``False``). The dataframes should
            have the same columns, which are the names of the methods.
        """
        self.dfs = dfs

    @abstractmethod
    def render(self, *args, **kwargs) -> Figure:
        raise NotImplementedError
