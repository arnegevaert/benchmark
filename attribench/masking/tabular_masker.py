from attribench.masking import Masker
import torch
import numpy as np
from typing import List, Union


class TabularMasker(Masker):
    """Basic Tabular masking class. Simply masks features by replacing them
    with a given constant value."""

    def __init__(self, mask_value: Union[float, List[float]] = 0.0):
        """
        Parameters
        ----------
        mask_value : Union[float, List[float]], optional
            The value to use for masking. By default 0.0.
        """
        super().__init__()
        self.mask_value = mask_value

    def _initialize_baselines(self, samples: torch.Tensor):
        self.baseline = (
            torch.ones(
                samples.shape, device=samples.device, dtype=samples.dtype
            )
            * self.mask_value
        )

    def set_batch(
        self, samples: torch.Tensor, attributions: torch.Tensor | None = None
    ):
        # Check if attributions and samples are compatible
        if attributions is not None and not self._check_attribution_shape(
            samples, attributions
        ):
            raise ValueError(
                "Attributions and samples have incompatible shapes."
            )

        # Set attributes
        self.samples = samples
        self.attributions = attributions
        if attributions is not None:
            self.sorted_indices = torch.tensor(
                attributions.cpu()
                .numpy()
                .reshape(attributions.shape[0], -1)
                .argsort()
            )

        # Init baselines
        self._initialize_baselines(samples)

    def _check_attribution_shape(self, samples, attributions) -> bool:
        # For tabular data, attributions and samples should always have the
        # same shape
        return samples.shape == attributions.shape
   
    def _mask(self, indices: torch.Tensor) -> torch.Tensor:
        if self.baseline is None:
            raise ValueError("Masker was not initialized.")

        assert self.samples is not None
        batch_size = self.samples.shape[0]
        num_indices = indices.shape[1]
        batch_dim = np.tile(range(batch_size), (num_indices, 1)).transpose()

        to_mask = torch.zeros(self.samples.shape).flatten(1)
        to_mask[batch_dim, indices] = 1.0
        return self._mask_boolean(to_mask.view(self.samples.shape).bool())

    def _mask_boolean(self, bool_mask: torch.Tensor) -> torch.Tensor:
        assert self.samples is not None
        return (
            self.samples
            - (bool_mask * self.samples)
            + (bool_mask * self.baseline)
        )
