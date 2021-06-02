from attrbench.metrics import MaskerActivationMetricResult
import pandas as pd
from typing import Tuple
import numpy as np


class DeletionResult(MaskerActivationMetricResult):
    inverted = True

    def get_df(self, mode="raw", include_baseline=False, masker: str = "constant",
               activation_fn: str = "linear", columns=None, agg_fn="aoc") -> Tuple[pd.DataFrame, bool]:
        if agg_fn not in ("aoc", "auc"):
            raise ValueError("agg_fn must be aoc or auc")
        pass  # TODO




class IrofResult(MaskerActivationMetricResult):
    inverted = True
