import numpy as np
import torch

from attrbench.lib import segment_samples, AttributionWriter
from attrbench.lib.masking import Masker



class _SensitivityNDataset:
    def __init__(self, n_range: np.ndarray, num_subsets: int, samples: torch.tensor,attrs: np.ndarray, masker: Masker):

        self.n_range = n_range
        self.num_subsets = num_subsets
        self.samples = samples
        self.masker = masker
        self.masker.initialize_batch(samples,attrs)
        # self.masker.initialize_baselines(samples) # can be used instead of init_batch here?
        self.rng = np.random.default_rng()

    def __len__(self):
        return self.n_range.shape[0] * self.num_subsets

    def __getitem__(self, item):
        n = self.n_range[item // self.num_subsets]
        masked_samples,indices= self.masker.mask_rand(n, True)
        return masked_samples, indices, n


class _SegSensNDataset(_SensitivityNDataset):
    def __init__(self, n_range: np.ndarray, num_subsets: int, samples: torch.tensor,attrs: np.ndarray,
                 masker: Masker, writer: AttributionWriter = None):
        super().__init__(n_range, num_subsets, samples,attrs,masker=masker)
        self.segmented_images = segment_samples(samples.cpu().numpy())
        self.attrs = attrs
        self.masker.initialize_batch(samples, attrs,segmented_samples = self.segmented_images)

        if writer is not None:
            writer.add_images("segmented samples", self.segmented_images)

    def __len__(self):
        return self.n_range.shape[0] * self.num_subsets

    def __getitem__(self, item):
        if self.masker is None:
            raise ValueError("Masker not set")
        n = self.n_range[item // self.num_subsets]

        masked_samples, indices = self.masker.mask_rand(n, True)
        return masked_samples, indices, n

    def set_masker(self, masker: Masker):
        self.masker = masker
        self.masker.initialize_baselines(self.samples, self.attrs,segmented_samples = self.segmented_images)
