from torch import nn
from abc import abstractmethod
import copy


class ModelFactory:
    """
    Basic interface for a callable that returns a model.
    This is necessary for multi-GPU mode, as each subprocess
    needs its own copy of the model.
    """

    @abstractmethod
    def __call__(self) -> nn.Module:
        """Return a model.

        Returns
        -------
        nn.Module
            The model.
        """
        raise NotImplementedError


class BasicModelFactory(ModelFactory):
    """
    Basic implementation of a :class:`ModelFactory` that just returns a deep copy
    of a given model. This can be used for simple models, but requires
    the model to be picklable (e.g. lambda layers won't work and require
    a specific implementation of :class:`ModelFactory`)
    """

    def __init__(self, base_model: nn.Module) -> None:
        """
        Parameters
        ----------
        base_model : nn.Module
            Model of which to return a deep copy.
        """
        self.base_model = base_model

    def __call__(self) -> nn.Module:
        """Return a deep copy of the base model.

        Returns
        -------
        nn.Module
            Deep copy of the base model.
        """
        return copy.deepcopy(self.base_model)
