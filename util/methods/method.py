import numpy as np


class Method:
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray, target: np.ndarray) -> np.ndarray:
        raise NotImplementedError
