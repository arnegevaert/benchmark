import numpy as np
import torch


class Masker:
    def __init__(self):
        self.baseline = None
        self.samples = None
        self.attributions = None
        self.sorted_indices = None

    def get_total_features(self):
        return self.sorted_indices.shape[1]

    def mask_top(self, k):
        if k == 0:
            return self.samples
        else:
            return self._mask(self.samples, self.sorted_indices[:, -k:])

    def mask_bot(self, k):
        return self._mask(self.samples, self.sorted_indices[:, :k])

    def mask_rand(self, k, return_indices=False):
        if k == 0:
            return self.samples
        rng = np.random.default_rng()

        num_samples = self.samples.shape[0]
        num_features = self.get_total_features()

        indices = np.arange(num_features)
        indices = np.tile(indices, (num_samples, 1))
        rng.shuffle(indices, axis=1)
        indices = indices[:, :k]
        masked_samples = self._mask(self.samples, indices)
        if return_indices: return masked_samples, indices
        return masked_samples

    def mask_all(self):
        return self.baseline

    def _check_attribution_shape(self, samples, attributions):
        raise NotImplementedError

    def _mask(self, samples: np.ndarray, indices: np.ndarray):
        raise NotImplementedError

    def _mask_boolean(self, samples, bool_mask):
        raise NotImplementedError

    def initialize_batch(self, samples: torch.tensor, attributions: np.ndarray = None):
        raise NotImplementedError
