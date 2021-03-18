import numpy as np
from torch.utils.data import Dataset

from attrbench.lib import mask_segments, segment_samples, AttributionWriter
from attrbench.lib.masking import Masker


class _SensitivityNDataset(Dataset):
    def __init__(self, n_range: np.ndarray, num_subsets: int, samples: np.ndarray,attrs: np.ndarray, num_features: int, masker: Masker):
        self.n_range = n_range
        self.num_subsets = num_subsets
        self.samples = samples
        masker_constructor, masker_kwargs=masker
        self.masker = masker_constructor(samples, attrs,**masker_kwargs)
        self.num_features = num_features

    def __len__(self):
        return self.n_range.shape[0] * self.num_subsets

    def __getitem__(self, item):
        n = self.n_range[item // self.num_subsets]
        masked_samples,indices= self.masker.mask_rand(n, True)
        return masked_samples, indices, n


class _SegSensNDataset(_SensitivityNDataset):
    def __init__(self, n_range: np.ndarray, num_subsets: int, samples: np.ndarray,attrs: np.ndarray,
                 masker: Masker, writer: AttributionWriter = None):
        super().__init__(n_range, num_subsets, samples,attrs, num_features=100, masker=masker)
        self.segmented_images = segment_samples(samples)
        masker_constructor, masker_kwargs=masker
        self.masker = masker_constructor(samples, attrs,segmented_samples = self.segmented_images,**masker_kwargs)
        if writer is not None:
            writer.add_images("segmented samples", self.segmented_images)

    def __len__(self):
        return self.n_range.shape[0] * self.num_subsets

    def __getitem__(self, item):
        n = self.n_range[item // self.num_subsets]
        rng = np.random.default_rng(5)  # Unique seed for each item ensures no duplicate indices
        indices = np.stack([rng.choice(np.unique(self.segmented_images[i, ...]), size=n, replace=False)
                            for i in range(self.samples.shape[0])])

        masked_samples= mask_segments(self.samples, self.segmented_images, indices, self.masker)

        masked_samples2, indices2 = self.masker.mask_rand(n, True)
        assert ((masked_samples == masked_samples2).all())
        return masked_samples, indices, n


