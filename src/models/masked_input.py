import torch.nn as nn
import torch
import itertools


class MaskedInputLayer(nn.Module):
    def __init__(self, sample_shape, radius=10, mask_value=-0.4242):
        # sample_shape: (n_channels, n_rows, n_cols)
        super(MaskedInputLayer, self).__init__()
        self.mask = torch.zeros(sample_shape)
        midpoint = (sample_shape[1]//2, sample_shape[2]//2)
        indices = [(i, j)
                   for (i, j) in itertools.product(range(sample_shape[1]), range(sample_shape[2]))
                   if abs(i - midpoint[0]) + abs(j - midpoint[1]) <= radius]
        self.mask[:, [i for i,j in indices], [j for i,j in indices]] = 1
        self.inverted_mask = torch.ones_like(self.mask) - self.mask
        self.mask_value = mask_value

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        return x * self.mask + self.inverted_mask * self.mask_value
