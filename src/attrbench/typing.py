from typing import NewType, Callable
import torch

AttributionMethod = NewType("AttributionMethod", Callable[[torch.Tensor, torch.Tensor], torch.Tensor])
