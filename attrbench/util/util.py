from typing import Dict, Callable
from torch.nn.functional import softmax
from torch import sigmoid
import torch


ACTIVATION_FNS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "linear": lambda x: x,
    "softmax": lambda x: softmax(x, dim=1),
    "sigmoid": lambda x: sigmoid(x)
}
