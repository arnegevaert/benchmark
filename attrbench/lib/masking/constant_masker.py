from attrbench.lib.masking import Masker
import numpy as np


class ConstantMasker(Masker):
    def __init__(self, feature_level, mask_value=0.):
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
