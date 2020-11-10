import numpy as np


class MaskingPolicy:
    def __call__(self, samples, indices):
        raise NotImplementedError


class FeatureMaskingPolicy(MaskingPolicy):
    def __init__(self, mask_value) -> None:
        super().__init__()
        self.mask_value = mask_value

    def __call__(self, samples, indices):
        flattened = samples.flatten(1)
        batch_dim = np.tile(range(samples.shape[0]), (indices.shape[1], 1)).transpose()
        flattened[batch_dim, indices] = self.mask_value
        return flattened.reshape(samples.shape)


class PixelMaskingPolicy(MaskingPolicy):
    def __init__(self, mask_value) -> None:
        super().__init__()
        self.mask_value = mask_value

    def __call__(self, samples, indices):
        flattened = samples.flatten(2)
        batch_dim = np.tile(range(samples.shape[0]), (indices.shape[1], 1)).transpose()
        flattened[batch_dim, :, indices] = self.mask_value
        return flattened.reshape(samples.shape)