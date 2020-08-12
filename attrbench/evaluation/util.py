import torch.nn.functional as F


transform_fns = {
    "identity": lambda l: l,
    "softmax": lambda l: F.softmax(l, dim=1),
}
