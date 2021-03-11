import numpy as np
from torch.utils.data import Dataset

from attrbench.lib import mask_segments, segment_samples
from attrbench.lib.masking import Masker


class _SensitivityNDataset(Dataset):
    def __init__(self, n_range: np.ndarray, num_subsets: int, samples: np.ndarray, num_features: int, masker: Masker):
        self.n_range = n_range
        self.num_subsets = num_subsets
        self.samples = samples
        self.masker = masker
        self.masker.initialize_baselines(samples)
        self.num_features = num_features

    def __len__(self):
        return self.n_range.shape[0] * self.num_subsets

    def __getitem__(self, item):
        n = self.n_range[item // self.num_subsets]
        rng = np.random.default_rng(item)  # Unique seed for each item ensures no duplicate indices
        indices = np.tile(rng.choice(self.num_features, size=n, replace=False), (self.samples.shape[0], 1))
        return self.masker.mask(self.samples, indices), indices, n


class _SegSensNDataset(_SensitivityNDataset):
    def __init__(self, n_range: np.ndarray, num_subsets: int, samples: np.ndarray,
                 masker: Masker, writer=None):
        super().__init__(n_range, num_subsets, samples, num_features=100, masker=masker)
        self.segmented_images = segment_samples(samples)
        if writer is not None:
            writer.add_images("segmented samples", self.segmented_images)

    def __len__(self):
        return self.n_range.shape[0] * self.num_subsets

    def __getitem__(self, item):
        n = self.n_range[item // self.num_subsets]
        rng = np.random.default_rng(item)  # Unique seed for each item ensures no duplicate indices
        indices = np.stack([rng.choice(np.unique(self.segmented_images[i, ...]), size=n, replace=False)
                            for i in range(self.samples.shape[0])])
        return mask_segments(self.samples, self.segmented_images, indices, self.masker), indices, n


