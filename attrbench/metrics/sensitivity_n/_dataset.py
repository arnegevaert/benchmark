import numpy as np
import torch

from attrbench.lib import segment_samples, AttributionWriter
from attrbench.lib.masking import ImageMasker, Masker


class _SensitivityNDataset:
    def __init__(self, n_range: np.ndarray, num_subsets: int, samples: torch.tensor, masker: Masker):
        self.n_range = n_range
        self.num_subsets = num_subsets
        self.samples = samples
        self.masker = masker
        self.masker.set_batch(samples)
        self.rng = np.random.default_rng()

    def __len__(self):
        return self.n_range.shape[0] * self.num_subsets

    def __getitem__(self, item):
        n = self.n_range[item // self.num_subsets]
        masked_samples, indices = self.masker.mask_rand(n, return_indices=True)
        return masked_samples, indices, n


class _SegSensNDataset:
    def __init__(self, n_range: np.ndarray, num_subsets: int, samples: torch.tensor,
                 writer: AttributionWriter = None):
        self.n_range = n_range
        self.num_subsets = num_subsets
        self.samples = samples
        self.segmented_images = torch.tensor(segment_samples(samples.cpu().numpy()), device=self.samples.device)
        self.masker = None

        if writer is not None:
            writer.add_images("segmented samples", self.segmented_images)

    def __len__(self):
        return self.n_range.shape[0] * self.num_subsets

    def __getitem__(self, item):
        n = self.n_range[item // self.num_subsets]
        masked_samples, indices = self.masker.mask_rand(n, True)
        return masked_samples, indices, n

    def set_masker(self, masker: ImageMasker):
        self.masker = masker
        self.masker.set_batch(self.samples, segmented_samples=self.segmented_images)
