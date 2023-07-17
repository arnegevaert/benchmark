from attribench.masking.image import ImageMasker
import torch
import numpy as np
from typing import List, Union


# TODO this should not be a subclass of ImageMasker
class TabularMasker(ImageMasker):
    def __init__(self, mask_value: Union[float, List[float]] = 0.0):
        self.mask_value = mask_value
        super().__init__("channel")

    def _initialize_baselines(self, samples):
        mask = torch.tensor(
            self.mask_value, device=samples.device, dtype=samples.dtype
        )
        self.baseline = (
            torch.ones(
                samples.shape, device=samples.device, dtype=samples.dtype
            )
            * mask
        )

    def _check_attribution_shape(self, samples, attributions):
        check1 = super()._check_attribution_shape(samples, attributions)
        if not isinstance(self.mask_value, float):
            return check1 and len(self.mask_value) == attributions.shape[-1]
        else:
            return check1

    def _mask(self, indices: np.ndarray):
        if self.baseline is None:
            raise ValueError("Masker was not initialized.")
        batch_size = self.samples.shape[0]
        num_indices = indices.shape[1]
        batch_dim = np.tile(range(batch_size), (num_indices, 1)).transpose()

        # to_mask = torch.zeros(samples.shape).flatten(1 if self.feature_level == "channel" else 2)
        to_mask = np.zeros(self.samples.shape)
        if self.masking_level == "channel":
            to_mask = to_mask.reshape((to_mask.shape[0], -1))
        else:
            to_mask = to_mask.reshape((to_mask.shape[0], to_mask.shape[1], -1))
        if self.masking_level == "channel":
            to_mask[batch_dim, indices] = 1.0
        else:
            try:
                to_mask[batch_dim, :, indices] = 1.0
            except IndexError:
                raise ValueError(
                    "Masking index was out of bounds. "
                    "Make sure the masking policy is compatible with method output."
                )
        to_mask = to_mask.reshape(self.samples.shape)
        return self._mask_boolean(to_mask)
