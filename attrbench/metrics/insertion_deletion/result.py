from typing import Tuple

import pandas as pd

from attrbench.metrics import MaskerActivationMetricResult
import numpy as np


class InsertionDeletionResult(MaskerActivationMetricResult):
    def get_df(self, **kwargs) -> Tuple[pd.DataFrame, bool]:
        return pd.DataFrame.from_dict(self.tree.get(postproc_fn=lambda x: np.trapz(x, x=np.linspace(0, 1, x.shape[-1])), **kwargs)), self.inverted


class InsertionResult(InsertionDeletionResult):
    inverted = False


class DeletionResult(InsertionDeletionResult):
    inverted = True


class IrofResult(MaskerActivationMetricResult):
    inverted = True


class IiofResult(MaskerActivationMetricResult):
    inverted = False
