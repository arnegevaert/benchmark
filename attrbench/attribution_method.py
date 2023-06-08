from abc import abstractmethod
import torch
from torch import nn


class AttributionMethod:
    """
    Wrapper class for attribution methods.
    An attribution method takes 2 arguments (input and target) and produces
    attributions in the shape of the input.
    To create a compatible attribution method, override the __call__ method
    and optionally the __init__ method for kwargs.
    """
    def __init__(self, model: nn.Module, **kwargs) -> None:
        self.model = model

    @abstractmethod
    def __call__(self, batch_x: torch.Tensor,
                 batch_target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
