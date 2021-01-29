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
        assert (blurred.shape == shape)

        res1 = channel_masker.mask(images, indices)
        res2 = pixel_masker.mask(images, indices)
        for i in range(shape[0]):
            for j in range(shape[-1] * shape[-2]):
                index = get_index(shape[1:], j)
                if j in indices[i]:
                    self.assertAlmostEqual(res1[i, index[0], index[1], index[2]].item(),
                                           blurred[i, index[0], index[1], index[2]].item())
                    self.assertAlmostEqual(res2[i, index[0], index[1], index[2]].item(),
                                           blurred[i, index[0], index[1], index[2]].item())
                else:
                    self.assertAlmostEqual(res1[i, index[0], index[1], index[2]].item(),
                                           images[i, index[0], index[1], index[2]].item())
                    self.assertAlmostEqual(res2[i, index[0], index[1], index[2]].item(),
                                           images[i, index[0], index[1], index[2]].item())
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

        assert (blurred.shape == shape)

        res = pixel_masker.mask(images, indices)
        for i in range(shape[0]):
            for j in range(shape[-1] * shape[-2]):
                index = get_index(shape[2:], j)
                if j in indices[i]:
                    for c in range(3):
                        self.assertAlmostEqual(blurred[i, c, index[0], index[1]].item(),
                                               res[i, c, index[0], index[1]].item())
                else:
                    for c in range(3):
                        self.assertAlmostEqual(images[i, c, index[0], index[1]].item(),
                                               res[i, c, index[0], index[1]].item())
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
        for i in range(shape[0]):
            for j in range(shape[-1] * shape[-2] * shape[-3]):
                index = get_index(shape[1:], j)
                if j in indices[i]:
                    self.assertAlmostEqual(blurred[i, index[0], index[1], index[2]].item(),
                                           res[i, index[0], index[1], index[2]].item())
                else:
                    self.assertAlmostEqual(images[i, index[0], index[1], index[2]].item(),
                                           res[i, index[0], index[1], index[2]].item())
        # Check that original images weren't mutated
        self.assertAlmostEqual((original - images).abs().sum().item(), 0.)
