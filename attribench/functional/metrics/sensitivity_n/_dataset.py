import numpy as np
from numpy import typing as npt
import torch

from attribench.masking import Masker
from attribench.masking.image import ImageMasker
from attribench._segmentation import segment_samples


class SensitivityNDataset:
    def __init__(
        self,
        n_range: npt.NDArray[np.int32],
        num_subsets: int,
        samples: torch.Tensor,
        masker: Masker,
    ):
        self.n_range = n_range
        self.num_subsets = num_subsets
        self.samples = samples
        self.masker = masker
        self.masker.set_batch(samples)
        self.rng = np.random.default_rng()

    def __len__(self):
        return self.n_range.shape[0] * self.num_subsets

    def __getitem__(self, item):
        # This is implicitly a nested loop, running through num_subsets and
        # n_range. The inner loop is the num_subsets loop, the outer loop is
        # the n_range loop.
        n_idx = item // self.num_subsets
        subset_idx = item % self.num_subsets
        masked_samples, indices = self.masker.mask_rand(
            int(self.n_range[n_idx]), return_indices=True
        )
        return masked_samples, indices, self.n_range[n_idx], subset_idx


class SegSensNDataset:
    def __init__(
        self,
        n_range: npt.NDArray[np.int32],
        num_subsets: int,
        samples: torch.Tensor,
    ):
        self.n_range = n_range
        self.num_subsets = num_subsets
        self.samples = samples
        self.segmented_images = torch.tensor(
            segment_samples(samples.cpu().numpy()), device=self.samples.device
        )
        self.masker: ImageMasker | None = None

    def __len__(self):
        return self.n_range.shape[0] * self.num_subsets

    def __getitem__(self, item):
        assert self.masker is not None
        n_idx = item // self.num_subsets
        subset_idx = item % self.num_subsets
        masked_samples, indices = self.masker.mask_rand(
            int(self.n_range[n_idx]), return_indices=True
        )
        return masked_samples, indices, self.n_range[n_idx], subset_idx

    def set_masker(self, masker: ImageMasker):
        self.masker = masker
        self.masker.set_batch(
            self.samples, segmented_samples=self.segmented_images
        )
