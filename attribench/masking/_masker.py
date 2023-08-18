from abc import abstractmethod
from typing import Tuple
import numpy as np
import torch


class Masker:
    """Base class for all maskers.
    Maskers are used to "remove" features from a sample by masking them with
    some value. This can be a fixed baseline value, a random value, or some
    other value.

    Note that a Masker object is not yet usable after creation. You need to
    call :meth:`set_batch` first, to set the samples and attributions.
    This allows the same Masker object to be used for multiple batches.
    """

    def __init__(self):
        self.baseline: torch.Tensor | None = None
        self.samples: torch.Tensor | None = None
        self.attributions: torch.Tensor | None = None
        self.sorted_indices: torch.Tensor | None = None
        self.rng = np.random.default_rng()

    def get_num_features(self) -> int:
        """Return the number of features in the samples.

        Returns
        -------
        int
            Number of features in the samples.
        """
        assert self.sorted_indices is not None
        return self.sorted_indices.shape[1]

    def mask_top(self, k: int) -> torch.Tensor:
        """Mask the ``k`` most important features, according to the attributions.

        Parameters
        ----------
        k : int
            Number of features to mask.

        Returns
        -------
        torch.Tensor
            Samples with the top k features masked.
        """
        assert self.sorted_indices is not None
        assert self.samples is not None
        if k == 0:
            return self.samples
        else:
            return self._mask(self.sorted_indices[:, -k:])

    def mask_bot(self, k: int) -> torch.Tensor:
        """Mask the ``k`` least important features, according to the attributions.

        Parameters
        ----------
        k : int
            Number of features to mask.

        Returns
        -------
        torch.Tensor
            Samples with the bottom k features masked.
        """
        assert self.sorted_indices is not None
        return self._mask(self.sorted_indices[:, :k])

    def mask_rand(
        self, k: int, return_indices=False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Mask ``k`` random features.

        Parameters
        ----------
        k : int
            Number of features to mask.
        return_indices : bool, optional
            Whether to return the indices of the masked features, by default False

        Returns
        -------
        torch.Tensor
            Samples with k random features masked.
        """
        assert self.samples is not None
        if k == 0:
            return self.samples

        num_samples = self.samples.shape[0]
        num_features = self.get_num_features()

        indices = torch.tensor(np.tile(
            self.rng.choice(num_features, size=k, replace=False),
            (num_samples, 1),
        ))
        masked_samples = self._mask(indices)
        if return_indices:
            return masked_samples, indices
        return masked_samples

    @abstractmethod
    def set_batch(
        self, samples: torch.Tensor, attributions: torch.Tensor | None = None
    ):
        """Set the samples and attributions for the next batch.

        Parameters
        ----------
        samples : torch.Tensor
            Samples of shape ``[num_samples, *sample_shape]``.
        attributions : torch.Tensor, optional
            Attributions of shape ``[num_samples, *sample_shape]``, by default None
            If None, the :meth:`mask_top` and :meth:`mask_bot` methods will 
            not be available.
        """
        raise NotImplementedError

    @abstractmethod
    def _check_attribution_shape(self, samples, attributions) -> bool:
        """Check if the attributions have the correct shape."""
        raise NotImplementedError

    @abstractmethod
    def _mask(self, indices: torch.Tensor) -> torch.Tensor:
        """Mask the given indices in the samples."""
        raise NotImplementedError

    @abstractmethod
    def _mask_boolean(self, bool_mask: torch.Tensor) -> torch.Tensor:
        """Mask using the given boolean mask."""
        raise NotImplementedError
