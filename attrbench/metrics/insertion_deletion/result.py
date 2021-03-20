from attrbench.metrics import ModeActivationMetricResult
from typing import Union, Dict
import pandas as pd
import numpy as np


class InsertionDeletionResult(ModeActivationMetricResult):
    def to_df(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        result = {}
        for mode in self.modes:
            for afn in self.activation_fn:
                df_dict = {}
                for m_name in self.method_names:
                    data = self.data[m_name][mode][afn].squeeze()
                    df_dict[m_name] = df_dict[m_name] = np.trapz(data, x=np.linspace(0, 1, data.shape[1]))
                result[f"{mode}_{afn}"] = pd.DataFrame.from_dict(df_dict)
        return result


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
