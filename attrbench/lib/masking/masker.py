import numpy as np
import torch
from typing import Optional


class Masker:
    def __init__(self):
        self.baseline: Optional[torch.Tensor] = None
        self.samples: Optional[torch.Tensor] = None
        self.attributions: Optional[np.ndarray] = None
        self.sorted_indices: Optional[np.ndarray] = None
        self.rng = np.random.default_rng()

    def set_batch(self, samples: torch.tensor, attributions: np.ndarray = None):
        raise NotImplementedError

    def get_num_features(self):
        return self.sorted_indices.shape[1]

    def mask_top(self, k):
        if k == 0:
            return self.samples
        else:
            return self._mask(self.sorted_indices[:, -k:])

    def mask_bot(self, k):
        return self._mask(self.sorted_indices[:, :k])

    def mask_rand(self, k, return_indices=False):
        if k == 0:
            return self.samples

        num_samples = self.samples.shape[0]
        num_features = self.get_num_features()

        indices = np.tile(self.rng.choice(num_features, size=k, replace=False), (num_samples, 1))
        masked_samples = self._mask(indices)
        if return_indices:
            return masked_samples, indices
        return masked_samples

    def _check_attribution_shape(self, samples, attributions):
        raise NotImplementedError

    def _mask(self, indices: np.ndarray):
        raise NotImplementedError

    def _mask_boolean(self, bool_mask):
        raise NotImplementedError

