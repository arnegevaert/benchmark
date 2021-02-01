import unittest
import torch
import numpy as np
from attrbench.lib import sum_of_attributions

def _get_index(shape, index):
    res = []
    for i in range(len(shape)):
        res.append(int(index // np.prod(shape[i+1:])))
        index = index % np.prod(shape[i+1:])
    return tuple(res)


class TestSumOfAttributions(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_sum_of_attributions(self):
        # Pixel-level (channel size 1) as well as channel-level (channel size 3)
        for shape, size in [((16, 1, 100, 100), 100), ((16, 3, 100, 100), 300)]:
            attrs = torch.randn(shape)
            indices = torch.stack([
                torch.tensor(
                    np.random.choice(shape[-1]*shape[-2]*shape[-3], size=size, replace=False)
                ) for _ in range(shape[0])
            ], dim=0)
            res = sum_of_attributions(attrs, indices)
            manual = torch.zeros(shape[0], 1)
            for i in range(shape[0]):
                for j in range(size):
                    index = _get_index(shape[1:], indices[i, j])
                    manual[i] += attrs[i, index[0], index[1], index[2]]
            diff = res - manual
            for i in range(shape[0]):
                self.assertAlmostEqual(diff[i, 0].item(), 0., places=4)