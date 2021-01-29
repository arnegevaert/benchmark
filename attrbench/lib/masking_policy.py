import numpy as np


class MaskingPolicy:
    def __call__(self, samples, indices):
        raise NotImplementedError

    def check_attribution_shape(self, samples, attributions):
        raise NotImplementedError


class FeatureMaskingPolicy(MaskingPolicy):
    def __init__(self, mask_value) -> None:
        super().__init__()
        self.mask_value = mask_value

    def __call__(self, samples, indices):
        flattened = samples.clone().flatten(1)
        batch_dim = np.tile(range(samples.shape[0]), (indices.shape[1], 1)).transpose()
        flattened[batch_dim, indices] = self.mask_value
        return flattened.reshape(samples.shape)

    def check_attribution_shape(self, samples, attributions):
        # FeatureMaskingPolicy expects attributions to have the same shape as samples
        return list(samples.shape) == list(attributions.shape)


class PixelMaskingPolicy(MaskingPolicy):
    def __init__(self, mask_value) -> None:
        super().__init__()
        self.mask_value = mask_value

    def __call__(self, samples, indices):
        flattened = samples.clone().flatten(2)
        batch_dim = np.tile(range(samples.shape[0]), (indices.shape[1], 1)).transpose()
        try:
            flattened[batch_dim, :, indices] = self.mask_value
        except IndexError:
            raise ValueError("Masking index was out of bounds. "
                             "Make sure the masking policy is compatible with method output.")
        return flattened.reshape(samples.shape)

    def check_attribution_shape(self, samples, attributions):
        # PixelMaskingPolicy expects attributions to have the same shape as samples,
        # except the channel dimension must be 1
        aggregated_shape = list(samples.shape)
        aggregated_shape[1] = 1
        return aggregated_shape == list(attributions.shape)