import unittest
import torch
import numpy as np
from attrbench.util import sum_of_attributions, mask_pixels, insert_pixels
    

def _get_index(shape, index):
    res = []
    for i in range(len(shape)):
        res.append(int(index // np.prod(shape[i+1:])))
        index = index % np.prod(shape[i+1:])
    return tuple(res)


class TestUtilMethodsRandomized(unittest.TestCase):
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
    
    def test_mask_pixels_grayscale(self):
        mask_value = 0.
        mask_size = 500
        shape = (16, 1, 100, 100)
        images = torch.randn(shape)
        indices = torch.stack([
            torch.tensor(
                np.random.choice(shape[-1]*shape[-2]*shape[-3], size=mask_size, replace=False)
            ) for _ in range(shape[0])
        ], dim=0)
        res1 = mask_pixels(images, indices, mask_value, pixel_level_mask=True)
        res2 = mask_pixels(images, indices, mask_value, pixel_level_mask=False)
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
        mask_size = 500
        shape = (16, 3, 100, 100)
        images = torch.randn(shape)
        indices = torch.stack([
            torch.tensor(
                np.random.choice(shape[-1]*shape[-2], size=mask_size, replace=False)
            ) for _ in range(shape[0])
        ], dim=0)
        res = mask_pixels(images, indices, mask_value, pixel_level_mask=True)
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
        mask_size = 500
        shape = (16, 3, 100, 100)
        images = torch.randn(shape)
        indices = torch.stack([
            torch.tensor(
                np.random.choice(shape[-1]*shape[-2]*shape[-3], size=mask_size, replace=False)
            ) for _ in range(shape[0])
        ], dim=0)
        res = mask_pixels(images, indices, mask_value, pixel_level_mask=False)
        manual = images.clone()
        for i in range(shape[0]):
            for j in range(mask_size):
                index = _get_index(shape[1:], indices[i, j])
                manual[i, index[0], index[1], index[2]] = mask_value
        diff = (res - manual).abs().sum().item()
        self.assertAlmostEqual(diff, 0., places=5)

    def test_insert_pixels_grayscale(self):
        mask_value = 0.
        insert_size = 500
        shape = (16, 1, 100, 100)
        images = torch.randn(shape)
        indices = torch.stack([
            torch.tensor(
                np.random.choice(shape[-1]*shape[-2]*shape[-3], size=insert_size, replace=False)
            ) for _ in range(shape[0])
        ], dim=0)
        res1 = insert_pixels(images, indices, mask_value, pixel_level_mask=True)
        res2 = insert_pixels(images, indices, mask_value, pixel_level_mask=False)
        manual = torch.ones(shape) * mask_value
        for i in range(shape[0]):
            for j in range(insert_size):
                index = _get_index(shape[1:], indices[i, j])
                manual[i, index[0], index[1], index[2]] = images[i, index[0], index[1], index[2]]
        diff1 = (res1 - manual).abs().sum().item()
        diff2 = (res2 - manual).abs().sum().item()
        self.assertAlmostEqual(diff1, 0.)
        self.assertAlmostEqual(diff2, 0.)

    def test_insert_pixels_rgb_pixel_level(self):
        mask_value = 0.
        insert_size = 500
        shape = (16, 3, 100, 100)
        images = torch.randn(shape)
        indices = torch.stack([
            torch.tensor(
                np.random.choice(shape[-1]*shape[-2], size=insert_size, replace=False)
            ) for _ in range(shape[0])
        ], dim=0)
        res = insert_pixels(images, indices, mask_value, pixel_level_mask=True)
        manual = torch.ones(shape) * mask_value
        for i in range(shape[0]):
            for j in range(3):
                for k in range(insert_size):
                    index = _get_index(shape[2:], indices[i,k])
                    manual[i, j, index[0], index[1]] = images[i, j, index[0], index[1]]
        diff = (res - manual).abs().sum().item()
        self.assertAlmostEqual(diff, 0.)

    def test_insert_pixels_rgb_channel_level(self):
        mask_value = 0.
        insert_size = 500
        shape = (16, 3, 100, 100)
        images = torch.randn(shape)
        indices = torch.stack([
            torch.tensor(
                np.random.choice(shape[-1]*shape[-2]*shape[-3], size=insert_size, replace=False)
            ) for _ in range(shape[0])
        ], dim=0)
        res = insert_pixels(images, indices, mask_value, pixel_level_mask=False)
        manual = torch.ones(shape) * mask_value
        for i in range(shape[0]):
            for j in range(insert_size):
                index = _get_index(shape[1:], indices[i,j])
                manual[i, index[0], index[1], index[2]] = images[i, index[0], index[1], index[2]]
        diff = (res - manual).abs().sum().item()
        self.assertAlmostEqual(diff, 0.)