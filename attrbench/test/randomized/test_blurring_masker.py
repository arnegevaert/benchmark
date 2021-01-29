import unittest
from attrbench.lib.masking import BlurringMasker
from attrbench.test.util import generate_images_indices, get_index
from cv2 import blur
import numpy as np
import torch


class TestBlurringMasker(unittest.TestCase):
    def _blur_images(self, images, kernel_size):
        blurred = []
        for i in range(images.shape[0]):
            cv_image = np.moveaxis(np.array(images[i, ...]), 0, 2)
            blurred_img = torch.tensor(blur(cv_image, (kernel_size, kernel_size)))
            if len(blurred_img.shape) == 2:
                blurred_img = blurred_img.unsqueeze(-1)
            blurred.append(blurred_img.permute(2, 0, 1))
        blurred = torch.stack(blurred, dim=0)
        return blurred

    def test_mask_pixels_grayscale(self):
        kernel_size = 3
        channel_masker = BlurringMasker("channel", kernel_size)
        pixel_masker = BlurringMasker("pixel", kernel_size)
        mask_size = 500
        shape = (16, 1, 100, 100)
        images, indices = generate_images_indices(shape, mask_size, "pixel")
        original = images.clone()
        blurred = self._blur_images(images, kernel_size)

        res1 = channel_masker.mask(images, indices)
        res2 = pixel_masker.mask(images, indices)
        manual = images.clone()
        for i in range(shape[0]):
            for j in range(mask_size):
                index = get_index(shape[1:], indices[i, j])
                manual[i, index[0], index[1], index[2]] = blurred[i, index[0], index[1], index[2]]
        # Check for correctness
        self.assertAlmostEqual((res1 - manual).abs().sum().item(), 0., places=5)
        self.assertAlmostEqual((res2 - manual).abs().sum().item(), 0., places=5)
        # Check that original images weren't mutated
        self.assertAlmostEqual((original - images).abs().sum().item(), 0.)

    def test_mask_pixels_rgb_pixel_level(self):
        kernel_size = 3
        pixel_masker = BlurringMasker("pixel", kernel_size)
        mask_size = 500
        shape = (16, 3, 100, 100)
        images, indices = generate_images_indices(shape, mask_size, "pixel")
        original = images.clone()
        blurred = self._blur_images(images, kernel_size)

        res = pixel_masker.mask(images, indices)
        manual = images.clone()
        for i in range(shape[0]):
            for j in range(3):
                for k in range(mask_size):
                    index = get_index(shape[2:], indices[i, k])
                    manual[i, j, index[0], index[1]] = blurred[i, j, index[0], index[1]]
        # Check for correctness
        self.assertAlmostEqual((res - manual).abs().sum().item(), 0., places=5)
        # Check that original images weren't mutated
        self.assertAlmostEqual((original - images).abs().sum().item(), 0.)

    def test_mask_pixels_channel_level(self):
        kernel_size = 3
        pixel_masker = BlurringMasker("channel", kernel_size)
        mask_size = 500
        shape = (16, 3, 100, 100)
        images, indices = generate_images_indices(shape, mask_size, "pixel")
        original = images.clone()
        blurred = self._blur_images(images, kernel_size)

        res = pixel_masker.mask(images, indices)
        manual = images.clone()
        for i in range(shape[0]):
            for j in range(mask_size):
                index = get_index(shape[1:], indices[i, j])
                manual[i, index[0], index[1], index[2]] = blurred[i, index[0], index[1], index[2]]
        # Check for correctness
        self.assertAlmostEqual((res - manual).abs().sum().item(), 0., places=5)
        # Check that original images weren't mutated
        self.assertAlmostEqual((original - images).abs().sum().item(), 0.)
