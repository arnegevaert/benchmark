from typing import NewType, Callable
import torch

Model = NewType("Model", Callable[[torch.Tensor], torch.Tensor])
AttributionMethod = NewType("AttributionMethod", Callable[[torch.Tensor, torch.Tensor], torch.Tensor])
