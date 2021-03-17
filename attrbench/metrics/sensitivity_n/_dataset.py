import numpy as np
import torch

from attrbench.lib import mask_segments, segment_samples, AttributionWriter
from attrbench.lib.masking import Masker


class _SensitivityNDataset:
    def __init__(self, n_range: np.ndarray, num_subsets: int, samples: torch.tensor, num_features: int, masker: Masker):
        self.n_range = n_range
        self.num_subsets = num_subsets
        self.samples = samples
        self.masker = masker
        self.masker.initialize_baselines(samples)
        self.num_features = num_features
        self.rng = np.random.default_rng()

    def __len__(self):
        return self.n_range.shape[0] * self.num_subsets

    def __getitem__(self, item):
        n = self.n_range[item // self.num_subsets]
        indices = np.tile(self.rng.choice(self.num_features, size=n, replace=False), (self.samples.shape[0], 1))
        return self.masker.mask(self.samples, indices), torch.tensor(indices), n


class _SegSensNDataset(_SensitivityNDataset):
    def __init__(self, n_range: np.ndarray, num_subsets: int, samples: torch.tensor,
                 masker: Masker, writer: AttributionWriter = None):
        super().__init__(n_range, num_subsets, samples, num_features=100, masker=masker)
        segmented_images = segment_samples(samples.cpu().numpy())
        self.segmented_images = torch.tensor(segmented_images, device=samples.device)
        self.segments = [np.unique(segmented_images[i, ...]) for i in range(samples.shape[0])]
        self.rng = np.random.default_rng()
        if writer is not None:
            writer.add_images("segmented samples", self.segmented_images)

    def __len__(self):
        return self.n_range.shape[0] * self.num_subsets

    def __getitem__(self, item):
        n = self.n_range[item // self.num_subsets]
        indices = torch.tensor(
            np.stack([self.rng.choice(self.segments[i], size=n, replace=False)
                      for i in range(self.samples.shape[0])]), device=self.samples.device)
        return mask_segments(self.samples, self.segmented_images, indices, self.masker), indices, n
