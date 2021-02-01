import unittest
import torch
import numpy as np
from attrbench.lib.masking import ConstantMasker
from os import path


class TestConstantMasker(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = {}
        for name in ("channel_level_attrs", "rgb_image", "grayscale_image", "pixel_level_attrs"):
            d = np.loadtxt(path.join(path.dirname(__file__), f"{name}.csv"), delimiter=",")
            self.data[name] = torch.tensor(d.reshape((2, -1, 3, 3))).float()
    
    def test_mask_pixels_grayscale(self):
        mask_value = 3.
        original = self.data["grayscale_image"].clone()
        fmp = ConstantMasker("channel", mask_value)
        pmp = ConstantMasker("pixel", mask_value)
        # Test masking for grayscale image
        # Use both pixel_level_mask=False and =True, should be exactly the same result
        indices = torch.tensor([[1, 5, 3], [2, 4, 6]])
        masked1 = fmp.mask(self.data["grayscale_image"], indices)
        masked2 = pmp.mask(self.data["grayscale_image"], indices)
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
        # Masking should not be in-place, original should be unchanged
        diff_orig = (original - self.data["grayscale_image"]).abs().sum().item()
        self.assertAlmostEqual(diff_orig, 0.)

    def test_mask_pixels_rgb_pixel_level(self):
        mask_value = 3.
        pmp = ConstantMasker("pixel", mask_value)
        indices = torch.tensor([[0, 5, 7], [2, 6, 1]])
        original = self.data["rgb_image"].clone()
        masked = pmp.mask(self.data["rgb_image"], indices)
        # Manual masking
        masked_manual = self.data["rgb_image"].clone()
        for i in range(3):
            masked_manual[0, i, 0, 0] = mask_value  # 0
            masked_manual[0, i, 1, 2] = mask_value  # 5
            masked_manual[0, i, 2, 1] = mask_value  # 7
            masked_manual[1, i, 0, 2] = mask_value  # 2
            masked_manual[1, i, 2, 0] = mask_value  # 6
            masked_manual[1, i, 0, 1] = mask_value  # 1
        diff = (masked_manual - masked).abs().sum().item()
        self.assertAlmostEqual(diff, 0.)
        # Masking should not be in-place, original should be unchanged
        diff_orig = (original - self.data["rgb_image"]).abs().sum().item()
        self.assertAlmostEqual(diff_orig, 0.)
    
    def test_mask_pixels_channel_level(self):
        mask_value = 3.
        fmp = ConstantMasker("channel", mask_value)
        indices = torch.tensor([[12, 21, 3], [5, 19, 25]])
        original = self.data["rgb_image"].clone()
        masked = fmp.mask(self.data["rgb_image"], indices)
        # Manual masking
        masked_manual = self.data["rgb_image"].clone()
        masked_manual[0, 1, 1, 0] = mask_value  # 12
        masked_manual[0, 2, 1, 0] = mask_value  # 21
        masked_manual[0, 0, 1, 0] = mask_value  # 3
        masked_manual[1, 0, 1, 2] = mask_value  # 5
        masked_manual[1, 2, 0, 1] = mask_value  # 19
        masked_manual[1, 2, 2, 1] = mask_value  # 25
        diff = (masked_manual - masked).abs().sum().item()
        self.assertAlmostEqual(diff, 0.)
        # Masking should not be in-place, original should be unchanged
        diff_orig = (original - self.data["rgb_image"]).abs().sum().item()
        self.assertAlmostEqual(diff_orig, 0.)