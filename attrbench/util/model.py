from torch import nn
import copy


class Model:
    """
    Wrapper class for a PyTorch model (nn.Module).
    When called, this class returns a copy of the base_model.
    This allows each subprocess to produce a copy of the model,
    which is necessary for multi-GPU applications.
    """
    def __init__(self, base_model: nn.Module) -> None:
        self.base_model = base_model

    def __call__(self) -> nn.Module:
        return copy.deepcopy(self.base_model)
