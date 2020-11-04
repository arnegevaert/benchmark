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
    
    def test_mask_pixels_grayscale(self):
        mask_value = 3.
        # Test masking for grayscale image
        # Use both pixel_level_mask=False and =True, should be exactly the same result
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

    def test_mask_pixels_rgb_pixel_level(self):
        mask_value = 3.
        indices = torch.tensor([[0, 5, 7], [2, 6, 1]])
        masked = mask_pixels(self.data["rgb_image"], indices, mask_value, pixel_level_mask=True)
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
    
    def test_mask_pixels_channel_level(self):
        mask_value = 3.
        indices = torch.tensor([[12, 21, 3], [5, 19, 25]])
        masked = mask_pixels(self.data["rgb_image"], indices, mask_value, pixel_level_mask=False)
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

    def test_insert_pixels_grayscale(self):
        mask_value = 3.
        # Use both pixel_level_mask=False and =True, should be exactly the same result
        indices = torch.tensor([[1, 5, 3], [2, 4, 6]])
        inserted1 = insert_pixels(self.data["grayscale_image"], indices, mask_value, pixel_level_mask=False)
        inserted2 = insert_pixels(self.data["grayscale_image"], indices, mask_value, pixel_level_mask=True)
        # Manual masking
        inserted_manual = torch.ones_like(self.data["grayscale_image"]) * mask_value
        inserted_manual[0, 0, 0, 1] = self.data["grayscale_image"][0, 0, 0, 1]  # 1
        inserted_manual[0, 0, 1, 2] = self.data["grayscale_image"][0, 0, 1, 2]  # 5
        inserted_manual[0, 0, 1, 0] = self.data["grayscale_image"][0, 0, 1, 0]  # 3
        inserted_manual[1, 0, 0, 2] = self.data["grayscale_image"][1, 0, 0, 2]  # 2
        inserted_manual[1, 0, 1, 1] = self.data["grayscale_image"][1, 0, 1, 1]  # 4
        inserted_manual[1, 0, 2, 0] = self.data["grayscale_image"][1, 0, 2, 0]  # 6
        # Calculate sum of absolute differences
        diff1 = (inserted_manual - inserted1).abs().sum().item()
        diff2 = (inserted_manual - inserted2).abs().sum().item()
        self.assertAlmostEqual(diff1, 0.)
        self.assertAlmostEqual(diff2, 0.)

    def test_insert_pixels_rgb_pixel_level(self):
        mask_value = 3.
        indices = torch.tensor([[0, 5, 7], [2, 6, 1]])
        inserted = insert_pixels(self.data["rgb_image"], indices, mask_value, pixel_level_mask=True)
        # Manual masking
        inserted_manual = torch.ones_like(self.data["rgb_image"]) * mask_value
        for i in range(3):
            inserted_manual[0, i, 0, 0] = self.data["rgb_image"][0, i, 0, 0]  # 0
            inserted_manual[0, i, 1, 2] = self.data["rgb_image"][0, i, 1, 2]  # 5
            inserted_manual[0, i, 2, 1] = self.data["rgb_image"][0, i, 2, 1]  # 7
            inserted_manual[1, i, 0, 2] = self.data["rgb_image"][1, i, 0, 2]  # 2
            inserted_manual[1, i, 2, 0] = self.data["rgb_image"][1, i, 2, 0]  # 6
            inserted_manual[1, i, 0, 1] = self.data["rgb_image"][1, i, 0, 1]  # 1
        diff = (inserted_manual - inserted).abs().sum().item()
        self.assertAlmostEqual(diff, 0.)

    def test_insert_pixels_rgb_channel_level(self):
        mask_value = 3.
        indices = torch.tensor([[12, 21, 3], [5, 19, 25]])
        inserted = insert_pixels(self.data["rgb_image"], indices, mask_value, pixel_level_mask=False)
        # Manual masking
        inserted_manual = torch.ones_like(self.data["rgb_image"]) * mask_value
        inserted_manual[0, 1, 1, 0] = self.data["rgb_image"][0, 1, 1, 0]  # 12
        inserted_manual[0, 2, 1, 0] = self.data["rgb_image"][0, 2, 1, 0]  # 21
        inserted_manual[0, 0, 1, 0] = self.data["rgb_image"][0, 0, 1, 0]  # 3
        inserted_manual[1, 0, 1, 2] = self.data["rgb_image"][1, 0, 1, 2]  # 5
        inserted_manual[1, 2, 0, 1] = self.data["rgb_image"][1, 2, 0, 1]  # 19
        inserted_manual[1, 2, 2, 1] = self.data["rgb_image"][1, 2, 2, 1]  # 25
        diff = (inserted_manual - inserted).abs().sum().item()
        self.assertAlmostEqual(diff, 0.)