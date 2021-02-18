import torch
import numpy as np


class Masker:
    def __init__(self, feature_level):
        if feature_level not in ("channel", "pixel"):
            raise ValueError(f"feature_level must be 'channel' or 'pixel'. Found {feature_level}.")
        self.feature_level = feature_level
        self.baseline = None

    def predict_masked(self, samples, indices, model, return_masked_samples=False):
        masked = self.mask(samples, indices)
        with torch.no_grad():
            pred = model(masked)
        if return_masked_samples:
            return pred, masked
        return pred

    def check_attribution_shape(self, samples, attributions):
        if self.feature_level == "channel":
            # Attributions should be same shape as samples
            return list(samples.shape) == list(attributions.shape)
        elif self.feature_level == "pixel":
            # attributions should have the same shape as samples,
            # except the channel dimension must be 1
            aggregated_shape = list(samples.shape)
            aggregated_shape[1] = 1
            return aggregated_shape == list(attributions.shape)

    def mask(self, samples, indices):
        if self.baseline is None:
            raise ValueError("Masker was not initialized.")
        batch_size, num_channels, rows, cols = samples.shape
        num_indices = indices.shape[1]
        batch_dim = np.tile(range(batch_size), (num_indices, 1)).transpose()

        to_mask = torch.zeros(samples.shape, device=samples.device).flatten(1 if self.feature_level == "channel" else 2)
        if self.feature_level == "channel":
            to_mask[batch_dim, indices] = 1.
        else:
            try:
                to_mask[batch_dim, :, indices] = 1.
            except IndexError:
                raise ValueError("Masking index was out of bounds. "
                                 "Make sure the masking policy is compatible with method output.")
        to_mask = to_mask.reshape(samples.shape)
        return self.mask_boolean(samples, to_mask)

    def mask_boolean(self, samples, bool_mask):
        bool_mask = bool_mask.to(samples.device)
        return samples - (bool_mask * samples) + (bool_mask * self.baseline.to(samples.device))

    def initialize_baselines(self, samples):
        raise NotImplementedError

