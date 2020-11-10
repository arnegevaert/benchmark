import unittest
import torch
import numpy as np
from attrbench.lib import mask_pixels
from attrbench.lib import FeatureMaskingPolicy, PixelMaskingPolicy

def _get_index(shape, index):
    res = []
    for i in range(len(shape)):
        res.append(int(index // np.prod(shape[i+1:])))
        index = index % np.prod(shape[i+1:])
    return tuple(res)


class TestMaskingRandomized(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def test_mask_pixels_grayscale(self):
        mask_value = 0.
        fmp = FeatureMaskingPolicy(mask_value)
        pmp = PixelMaskingPolicy(mask_value)
        mask_size = 500
        shape = (16, 1, 100, 100)
        images = torch.randn(shape)
        indices = torch.stack([
            torch.tensor(
                np.random.choice(shape[-1]*shape[-2]*shape[-3], size=mask_size, replace=False)
            ) for _ in range(shape[0])
        ], dim=0)
        #res1 = mask_pixels(images, indices, mask_value, pixel_level_mask=True)
        #res2 = mask_pixels(images, indices, mask_value, pixel_level_mask=False)
        res1 = fmp(images, indices)
        res2 = pmp(images, indices)
        manual = images.clone()
        for i in range(shape[0]):
            for j in range(mask_size):
                index = _get_index(shape[1:], indices[i, j])
                manual[i, index[0], index[1], index[2]] = mask_value
        diff1 = (res1 - manual).abs().sum().item()
        diff2 = (res2 - manual).abs().sum().item()
        self.assertAlmostEqual(diff1, 0., places=5)
        self.assertAlmostEqual(diff2, 0., places=5)

    def test_mask_pixels_rgb_pixel_level(self):
        mask_value = 0.
        pmp = PixelMaskingPolicy(mask_value)
        mask_size = 500
        shape = (16, 3, 100, 100)
        images = torch.randn(shape)
        indices = torch.stack([
            torch.tensor(
                np.random.choice(shape[-1]*shape[-2], size=mask_size, replace=False)
            ) for _ in range(shape[0])
        ], dim=0)
        #res = mask_pixels(images, indices, mask_value, pixel_level_mask=True)
        res = pmp(images, indices)
        manual = images.clone()
        for i in range(shape[0]):
            for j in range(3):
                for k in range(mask_size):
                    index = _get_index(shape[2:], indices[i, k])
                    manual[i, j, index[0], index[1]] = mask_value
        diff = (res - manual).abs().sum().item()
        self.assertAlmostEqual(diff, 0., places=5)
    
    def test_mask_pixels_channel_level(self):
        mask_value = 0.
        fmp = FeatureMaskingPolicy(mask_value)
        mask_size = 500
        shape = (16, 3, 100, 100)
        images = torch.randn(shape)
        indices = torch.stack([
            torch.tensor(
                np.random.choice(shape[-1]*shape[-2]*shape[-3], size=mask_size, replace=False)
            ) for _ in range(shape[0])
        ], dim=0)
        #res = mask_pixels(images, indices, mask_value, pixel_level_mask=False)
        res = fmp(images, indices)
        manual = images.clone()
        for i in range(shape[0]):
            for j in range(mask_size):
                index = _get_index(shape[1:], indices[i, j])
                manual[i, index[0], index[1], index[2]] = mask_value
        diff = (res - manual).abs().sum().item()
        self.assertAlmostEqual(diff, 0., places=5)