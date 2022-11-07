import numpy as np
from torch.nn.functional import softmax
from torch import sigmoid
import warnings
from collections import defaultdict



ACTIVATION_FNS = {
    "linear": lambda x: x,
    "softmax": lambda x: softmax(x, dim=1),
    "sigmoid": lambda x: sigmoid(x)
}