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
        if self.feature_level == "channel":
            # Attributions should be same shape as samples
            return list(samples.shape) == list(attributions.shape)
        elif self.feature_level == "pixel":
            # attributions should have the same shape as samples,
            # except the channel dimension must be 1
            aggregated_shape = list(samples.shape)
            aggregated_shape[1] = 1
            return aggregated_shape == list(attributions.shape)
