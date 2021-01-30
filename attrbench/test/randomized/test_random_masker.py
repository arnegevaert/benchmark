import unittest
from attrbench.lib.masking import RandomMasker
from attrbench.test.util import get_index, generate_images_indices
import torch
import numpy as np


class TestRandomMasker(unittest.TestCase):
    def test_mask_pixels_grayscale(self):
        channel_masker = RandomMasker("channel")
        pixel_masker = RandomMasker("pixel")
        mask_size = 500
        shape = (16, 1, 100, 100)
        images, indices = generate_images_indices(shape, mask_size, "pixel")
        original = images.clone()
        res1 = channel_masker.mask(images, indices)
        res2 = pixel_masker.mask(images, indices)
        for i in range(shape[0]):
            rand_indices = torch.tensor(np.random.choice(shape[-1]*shape[-2], size=mask_size, replace=False))
            for pert_idx in indices[i]:
                index = get_index(shape[1:], pert_idx)
                # This value should be different (replaced by random value)
                self.assertNotEqual(images[i, index[0], index[1], index[2]],
                                    res1[i, index[0], index[1], index[2]])
                self.assertNotEqual(images[i, index[0], index[1], index[2]],
                                    res2[i, index[0], index[1], index[2]])
            for rand_idx in rand_indices:
                index = get_index(shape[1:], rand_idx)
                if rand_idx in indices[i]:
                    # This value should be different (replaced by random value)
                    self.assertNotEqual(images[i, index[0], index[1], index[2]],
                                        res1[i, index[0], index[1], index[2]])
                    self.assertNotEqual(images[i, index[0], index[1], index[2]],
                                        res2[i, index[0], index[1], index[2]])
                else:
                    # This value should be the same
                    self.assertAlmostEqual(images[i, index[0], index[1], index[2]],
                                           res1[i, index[0], index[1], index[2]])
                    self.assertAlmostEqual(images[i, index[0], index[1], index[2]],
                                           res2[i, index[0], index[1], index[2]])
        # Check that original images weren't mutated
        self.assertAlmostEqual((original - images).abs().sum().item(), 0.)

    def test_mask_pixels_rgb_pixel_level(self):
        pixel_masker = RandomMasker("pixel")
        mask_size = 500
        shape = (16, 3, 100, 100)
        images, indices = generate_images_indices(shape, mask_size, "pixel")
        original = images.clone()
        res = pixel_masker.mask(images, indices)
        for i in range(shape[0]):
            rand_indices = torch.tensor(np.random.choice(shape[-1]*shape[-2], size=mask_size, replace=False))
            for pert_idx in indices[i]:
                index = get_index(shape[2:], pert_idx)
                for c in range(3):
                    # This value should be different (replaced by random value)
                    self.assertNotEqual(images[i, c, index[0], index[1]],
                                        res[i, c, index[0], index[1]])
            for rand_idx in rand_indices:
                index = get_index(shape[2:], rand_idx)
                if rand_idx in indices[i]:
                    for c in range(3):
                        # This value should be different (replaced by random value)
                        self.assertNotEqual(images[i, c, index[0], index[1]],
                                            res[i, c, index[0], index[1]])
                else:
                    for c in range(3):
                        # This value should be the same
                        self.assertAlmostEqual(images[i, c, index[0], index[1]],
                                               res[i, c, index[0], index[1]])
        # Check that original images weren't mutated
        self.assertAlmostEqual((original - images).abs().sum().item(), 0.)

    def test_mask_pixels_channel_level(self):
        pixel_masker = RandomMasker("channel")
        mask_size = 500
        shape = (16, 3, 100, 100)
        images, indices = generate_images_indices(shape, mask_size, "channel")
        original = images.clone()
        res = pixel_masker.mask(images, indices)
        for i in range(shape[0]):
            rand_indices = torch.tensor(np.random.choice(shape[-1]*shape[-2]*shape[-3], size=mask_size, replace=False))
            for pert_idx in indices[i]:
                index = get_index(shape[1:], pert_idx)
                # This value should be different (replaced by random value)
                self.assertNotEqual(images[i, index[0], index[1], index[2]],
                                    res[i, index[0], index[1], index[2]])
            for rand_idx in rand_indices:
                index = get_index(shape[1:], rand_idx)
                if rand_idx in indices[i]:
                    # This value should be different (replaced by random value)
                    self.assertNotEqual(images[i, index[0], index[1], index[2]],
                                        res[i, index[0], index[1], index[2]])
                else:
                    # This value should be the same
                    self.assertAlmostEqual(images[i, index[0], index[1], index[2]],
                                           res[i, index[0], index[1], index[2]])
        # Check that original images weren't mutated
        self.assertAlmostEqual((original - images).abs().sum().item(), 0.)
