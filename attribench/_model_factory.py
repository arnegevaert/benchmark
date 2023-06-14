from torch import nn
import copy


class ModelFactory:
    """
    Basic interface for a callable that returns a model.
    This is necessary for multi-GPU mode, as each subprocess
    needs its own copy of the model.
    """
    def __call__(self) -> nn.Module:
        raise NotImplementedError


class BasicModel(ModelFactory):
    """
    Basic implementation of a ModelFactory that just returns a deep copy
    of a given model. This can be used for simple models, but requires
    the model to be picklable (e.g. lambda layers won't work and require
    a specific implementation of ModelFactory)
    """
    def __init__(self, base_model: nn.Module) -> None:
        self.base_model = base_model

    def __call__(self) -> nn.Module:
        return copy.deepcopy(self.base_model)
