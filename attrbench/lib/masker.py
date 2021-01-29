import numpy as np
import torch


class Masker:
    def __init__(self, feature_level):
        if feature_level not in ("channel", "pixel"):
            raise ValueError(f"feature_level must be 'channel' or 'pixel'. Found {feature_level}.")
        self.feature_level = feature_level

    def predict_masked(self, samples, indices, model, return_masked_samples=False):
        masked = self.mask(samples, indices)
        with torch.no_grad():
            pred = model(masked)
        if return_masked_samples:
            return pred, masked
        return pred

    def mask(self, samples, indices):
        raise NotImplementedError

    def check_attribution_shape(self, samples, attributions):
        raise NotImplementedError


class ConstantMasker(Masker):
    def __init__(self, feature_level, mask_value=0):
        super().__init__(feature_level)
        self.mask_value = mask_value

    def mask(self, samples, indices):
        flattened = samples.clone().flatten(1 if self.feature_level == "channel" else 2)
        batch_dim = np.tile(range(samples.shape[0]), (indices.shape[1], 1)).transpose()

        if self.feature_level == "channel":
            flattened[batch_dim, indices] = self.mask_value
            return flattened.reshape(samples.shape)
        elif self.feature_level == "pixel":
            try:
                flattened[batch_dim, :, indices] = self.mask_value
            except IndexError:
                raise ValueError("Masking index was out of bounds. "
                                 "Make sure the masking policy is compatible with method output.")
            return flattened.reshape(samples.shape)

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


class SampleAverageMasker(Masker):
    def __init__(self, feature_level, mask_value=0):
        super().__init__(feature_level)
        self.mask_value = mask_value

    def mask(self, samples, indices):
        return samples

    def check_attribution_shape(self, samples, attributions):
        return True


class BlurringMasker(Masker):
    def mask(self, samples, indices):
        return samples

    def check_attribution_shape(self, samples, attributions):
        return True


class RandomMasker(Masker):
    def mask(self, samples, indices):
        return samples

    def check_attribution_shape(self, samples, attributions):
        return True
