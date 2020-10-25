import unittest
import torch
import numpy as np
from attrbench.util import sum_of_attributions, mask_pixels, insert_pixels
from os import path


class TestUtilMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = {}
        for name in ("channel_level_attrs", "rgb_image", "grayscale_image", "pixel_level_attrs"):
            d = np.loadtxt(path.join(path.dirname(__file__), f"{name}.csv"), delimiter=",")
            self.data[name] = torch.tensor(d.reshape((2, -1, 3, 3)))

    def test_sum_of_attributions(self):
        # Test sum of attributions for pixel level attributions
        pixel_indices = torch.tensor([[1, 3, 5, 6], [1, 3, 5, 4]])
        pixel_level = sum_of_attributions(self.data["pixel_level_attrs"], pixel_indices)
        self.assertTupleEqual(pixel_level.shape, (2, 1))
        self.assertAlmostEqual(pixel_level[0, 0], 1.2)
        self.assertAlmostEqual(pixel_level[1, 0], 1.1)

        # Test sum of attributions for channel level attributions
        channel_indices = torch.tensor([[1, 5, 12, 17], [1, 5, 12, 16]])
        channel_level = sum_of_attributions(self.data["channel_level_attrs"], channel_indices)
        self.assertTupleEqual(channel_level.shape, (2, 1))
        self.assertAlmostEqual(channel_level[0, 0], 1.4)
        self.assertAlmostEqual(channel_level[1, 0], 2.4)

    def test_mask_pixels_grayscale(self):
        mask_value = 3.
        # Test masking for grayscale image
        indices = torch.tensor([[1, 5, 3], [2, 4, 6]])
        masked1 = mask_pixels(self.data["grayscale_image"], indices, mask_value, pixel_level_mask=False)
        masked2 = mask_pixels(self.data["grayscale_image"], indices, mask_value, pixel_level_mask=True)
        # Manual masking
        masked_manual = self.data["grayscale_image"].clone()
        masked_manual[0, 0, 0, 1] = mask_value  # 1
        masked_manual[0, 0, 1, 2] = mask_value  # 5
        masked_manual[0, 0, 1, 0] = mask_value  # 3
        masked_manual[1, 0, 0, 2] = mask_value  # 2
        masked_manual[1, 0, 1, 1] = mask_value  # 4
        masked_manual[1, 0, 2, 0] = mask_value  # 6
        # Calculate sum of absolute differences
        diff1 = (masked_manual - masked1).abs().sum().item()
        diff2 = (masked_manual - masked2).abs().sum().item()
        self.assertAlmostEqual(diff1, 0.)
        self.assertAlmostEqual(diff2, 0.)
    
    def test_mask_pixels_rgb(self):
        self.fail("Not implemented")

    def test_insert_pixels_grayscale(self):
        self.fail("Not implemented")
    
    def test_insert_pixels_rgb(self):
        self.fail("Not implemented")