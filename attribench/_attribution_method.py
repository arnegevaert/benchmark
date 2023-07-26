from abc import abstractmethod
import torch
from torch import nn


class AttributionMethod:
    """
    Wrapper class for attribution methods.
    An attribution method takes 2 arguments (input and target) and produces
    attributions in the shape of the input.
    To create a compatible attribution method, override the :meth:`__call__` method
    and optionally the :meth:`__init__` method for kwargs.
    """
    def __init__(self, model: nn.Module, **kwargs) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            Model to compute attributions for.
        """
        self.model = model

    @abstractmethod
    def __call__(self, batch_x: torch.Tensor,
                 batch_target: torch.Tensor) -> torch.Tensor:
        """Compute attributions for a batch of inputs.

        Parameters
        ----------
        batch_x : torch.Tensor
            Input samples.
        batch_target : torch.Tensor
            Targets to compute attributions for.
            Note that these need not be the same as the ground truth targets.

        Returns
        -------
        torch.Tensor
            Attributions for the given inputs and targets.

        """
        raise NotImplementedError
