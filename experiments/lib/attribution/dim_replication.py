import torch
from typing import Callable

class DimReplication:
    def __init__(self, base_method: Callable, dim: int, amount: int) -> None:
        self.base_method = base_method
        self.dim = dim
        self.amount = amount

    def __call__(self, x, target):
        attrs = self.base_method(x, target)
        shape = [1 for _ in range(len(attrs.shape))]
        shape[self.dim] = self.amount
        shape = tuple(shape)
        return attrs.repeat(*shape)