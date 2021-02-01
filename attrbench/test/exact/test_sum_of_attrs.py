import unittest
from os import path
import torch
from attrbench.lib import sum_of_attributions
import numpy as np

class TestSumOfAttributions(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = {}
        for name in ("channel_level_attrs", "pixel_level_attrs"):
            d = np.loadtxt(path.join(path.dirname(__file__), f"{name}.csv"), delimiter=",")
            self.data[name] = torch.tensor(d.reshape((2, -1, 3, 3))).float()

    def test_sum_of_attributions_pixel_level(self):
        # Test sum of attributions for pixel level attributions
        pixel_indices = torch.tensor([[1, 3, 5, 6], [1, 3, 5, 4]])
        res = sum_of_attributions(self.data["pixel_level_attrs"], pixel_indices)
        self.assertTupleEqual(res.shape, (2, 1))
        self.assertAlmostEqual(res[0, 0], 1.2)
        self.assertAlmostEqual(res[1, 0], 1.1)

    def test_sum_of_attributions_channel_level(self):
        # Test sum of attributions for channel level attributions
        channel_indices = torch.tensor([[1, 5, 12, 17], [1, 5, 12, 16]])
        channel_level = sum_of_attributions(self.data["channel_level_attrs"], channel_indices)
        self.assertTupleEqual(channel_level.shape, (2, 1))
        self.assertAlmostEqual(channel_level[0, 0].item(), 1.4, places=6)
        self.assertAlmostEqual(channel_level[1, 0].item(), 2.4, places=6)