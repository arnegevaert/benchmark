from abc import abstractmethod
import numpy as np
import torch


class Masker:
    def __init__(self):
        self.baseline: torch.Tensor | None = None
        self.samples: torch.Tensor | None = None
        self.attributions: torch.Tensor | None = None
        self.sorted_indices: torch.Tensor | None = None
        self.rng = np.random.default_rng()

    def get_num_features(self):
        assert self.sorted_indices is not None
        return self.sorted_indices.shape[1]

    def mask_top(self, k):
        assert self.sorted_indices is not None
        if k == 0:
            return self.samples
        else:
            return self._mask(self.sorted_indices[:, -k:])

    def mask_bot(self, k):
        assert self.sorted_indices is not None
        return self._mask(self.sorted_indices[:, :k])

    def mask_rand(self, k, return_indices=False):
        assert self.samples is not None
        if k == 0:
            return self.samples

        num_samples = self.samples.shape[0]
        num_features = self.get_num_features()

        indices = np.tile(
            self.rng.choice(num_features, size=k, replace=False),
            (num_samples, 1),
        )
        masked_samples = self._mask(torch.tensor(indices))
        if return_indices:
            return masked_samples, indices
        return masked_samples

    @abstractmethod
    def set_batch(
        self, samples: torch.Tensor, attributions: torch.Tensor | None = None
    ):
        raise NotImplementedError

    @abstractmethod
    def _check_attribution_shape(self, samples, attributions):
        raise NotImplementedError

    @abstractmethod
    def _mask(self, indices: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def _mask_boolean(self, bool_mask):
        raise NotImplementedError
