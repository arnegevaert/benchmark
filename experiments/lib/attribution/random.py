import torch

class Random:
    def __call__(self, x, target):
        return (torch.rand(*x.shape) * 2 - 1).to(x.device)