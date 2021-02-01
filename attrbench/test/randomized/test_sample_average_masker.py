import unittest
import torch
from attrbench.lib.masking import SampleAverageMasker
from attrbench.test.util import get_index, generate_images_indices


class TestSampleAverageMasker(unittest.TestCase):
    def test_mask_pixels_grayscale(self):
        channel_masker = SampleAverageMasker("channel")
        pixel_masker = SampleAverageMasker("pixel")
        mask_size = 500
        shape = (16, 1, 100, 100)
        images, indices = generate_images_indices(shape, mask_size, "pixel")
        original = images.clone()
        res1 = channel_masker.mask(images, indices)
        res2 = pixel_masker.mask(images, indices)
        manual = images.clone()
        for i in range(shape[0]):
            mask_value = torch.mean(images[i, ...]).item()
            for j in range(mask_size):
                index = get_index(shape[1:], indices[i, j])
                manual[i, index[0], index[1], index[2]] = mask_value
        # Check for correctness
        self.assertAlmostEqual((res1 - manual).abs().sum().item(), 0., places=5)
        self.assertAlmostEqual((res2 - manual).abs().sum().item(), 0., places=5)
        # Check that original images weren't mutated
        self.assertAlmostEqual((original - images).abs().sum().item(), 0.)

    def test_mask_pixels_rgb_pixel_level(self):
        masker = SampleAverageMasker("pixel")
        mask_size = 500
        shape = (16, 3, 100, 100)
        images, indices = generate_images_indices(shape, mask_size, "pixel")
        original = images.clone()
        res = masker.mask(images, indices)
        manual = images.clone()
        for i in range(shape[0]):
            for j in range(3):
                mask_value = torch.mean(images[i, j, ...]).item()
                for k in range(mask_size):
                    index = get_index(shape[2:], indices[i, k])
                    manual[i, j, index[0], index[1]] = mask_value
        # Check for correctness
        self.assertAlmostEqual((res - manual).abs().sum().item(), 0., places=5)
        # Check that original images weren't mutated
        self.assertAlmostEqual((original - images).abs().sum().item(), 0.)

    def test_mask_pixels_channel_level(self):
        masker = SampleAverageMasker("channel")
        mask_size = 500
        shape = (16, 3, 100, 100)
        images, indices = generate_images_indices(shape, mask_size, "channel")
        original = images.clone()
        res = masker.mask(images, indices)
        manual = images.clone()
        for i in range(shape[0]):
            for j in range(mask_size):
                index = get_index(shape[1:], indices[i, j])
                channel = index[0]
                mask_value = torch.mean(images[i, channel, ...]).item()
                manual[i, index[0], index[1], index[2]] = mask_value
        # Check for correctness
        self.assertAlmostEqual((res - manual).abs().sum().item(), 0., places=5)
        # Check that original images weren't mutated
        self.assertAlmostEqual((original - images).abs().sum().item(), 0.)
