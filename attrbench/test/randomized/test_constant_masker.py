import unittest
import torch
import numpy as np
from attrbench.lib import ConstantMasker
from attrbench.test.util import get_index, generate_images_indices


class TestConstantMasker(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def test_mask_pixels_grayscale(self):
        mask_value = 0.
        fmp = ConstantMasker("channel")
        pmp = ConstantMasker("pixel")
        mask_size = 500
        shape = (16, 1, 100, 100)
        images, indices = generate_images_indices(shape, mask_size, "pixel")
        original = images.clone()
        res1 = fmp.mask(images, indices)
        res2 = pmp.mask(images, indices)
        manual = images.clone()
        for i in range(shape[0]):
            for j in range(mask_size):
                index = get_index(shape[1:], indices[i, j])
                manual[i, index[0], index[1], index[2]] = mask_value
        # Check for correctness
        self.assertAlmostEqual((res1 - manual).abs().sum().item(), 0., places=5)
        self.assertAlmostEqual((res2 - manual).abs().sum().item(), 0., places=5)
        # Check that original images weren't mutated
        self.assertAlmostEqual((original - images).abs().sum().item(), 0.)

    def test_mask_pixels_rgb_pixel_level(self):
        mask_value = 0.
        pmp = ConstantMasker("pixel")
        mask_size = 500
        shape = (16, 3, 100, 100)
        images, indices = generate_images_indices(shape, mask_size, "pixel")
        original = images.clone()
        res = pmp.mask(images, indices)
        manual = images.clone()
        for i in range(shape[0]):
            for j in range(3):
                for k in range(mask_size):
                    index = get_index(shape[2:], indices[i, k])
                    manual[i, j, index[0], index[1]] = mask_value
        # Check for correctness
        self.assertAlmostEqual((res - manual).abs().sum().item(), 0., places=5)
        # Check that original images weren't mutated
        self.assertAlmostEqual((original - images).abs().sum().item(), 0.)
    
    def test_mask_pixels_channel_level(self):
        mask_value = 0.
        fmp = ConstantMasker("channel")
        mask_size = 500
        shape = (16, 3, 100, 100)
        images, indices = generate_images_indices(shape, mask_size, "channel")
        original = images.clone()
        res = fmp.mask(images, indices)
        manual = images.clone()
        for i in range(shape[0]):
            for j in range(mask_size):
                index = get_index(shape[1:], indices[i, j])
                manual[i, index[0], index[1], index[2]] = mask_value
        # Check for correctness
        self.assertAlmostEqual((res - manual).abs().sum().item(), 0., places=5)
        # Check that original images weren't mutated
        self.assertAlmostEqual((original - images).abs().sum().item(), 0.)