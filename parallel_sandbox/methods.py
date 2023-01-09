from attrbench.util.attribution_method import AttributionMethod
import torch
from torch import nn
from captum import attr


class Gradient(AttributionMethod):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        self.saliency = attr.Saliency(self.model)

    def __call__(self, batch_x: torch.Tensor,
                 batch_target: torch.Tensor) -> torch.Tensor:
        return self.saliency.attribute(batch_x, batch_target)


class InputXGradient(AttributionMethod):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        self.ixg = attr.InputXGradient(self.model)

    def __call__(self, batch_x: torch.Tensor,
                 batch_target: torch.Tensor) -> torch.Tensor:
        return self.ixg.attribute(batch_x, batch_target)


class IntegratedGradients(AttributionMethod):
    def __init__(self, model: nn.Module, batch_size: int) -> None:
        super().__init__(model)
        self.integrated_gradients = attr.IntegratedGradients(self.model)
        self.batch_size = batch_size

    def __call__(self, batch_x: torch.Tensor,
                 batch_target: torch.Tensor) -> torch.Tensor:
        return self.integrated_gradients.attribute(
                inputs=batch_x,
                target=batch_target,
                internal_batch_size=self.batch_size)


class Random(AttributionMethod):
    def __call__(self, batch_x: torch.Tensor,
                 batch_target: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(batch_x)
