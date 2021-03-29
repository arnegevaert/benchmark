import numpy as np
import torch

from attrbench.lib import segment_samples, AttributionWriter
from attrbench.lib.masking import Masker



class _SensitivityNDataset:
    def __init__(self, n_range: np.ndarray, num_subsets: int, samples: torch.tensor,attrs: np.ndarray, num_features: int, masker: Masker):

        self.n_range = n_range
        self.num_subsets = num_subsets
        self.samples = samples
        masker_constructor, masker_kwargs=masker
        self.masker = masker_constructor(samples, attrs,**masker_kwargs)
        self.num_features = num_features
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
        super().__init__(n_range, num_subsets, samples,attrs, num_features=100, masker=masker)
        self.segmented_images = segment_samples(samples.cpu().numpy())
        masker_constructor, masker_kwargs=masker
        self.masker = masker_constructor(samples, attrs,segmented_samples = self.segmented_images,**masker_kwargs)
        if writer is not None:
            writer.add_images("segmented samples", self.segmented_images)

    def __len__(self):
        return self.n_range.shape[0] * self.num_subsets

    def __getitem__(self, item):
        n = self.n_range[item // self.num_subsets]
        masked_samples, indices = self.masker.mask_rand(n, True)
        return masked_samples, indices, n

