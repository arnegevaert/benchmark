from attrbench.lib.masking import Masker
import numpy as np
from cv2 import blur
import torch


class BlurringMasker(Masker):
    def __init__(self, feature_level, kernel_size):
        super().__init__(feature_level)
        self.kernel_size = kernel_size

    def mask(self, samples, indices):
        res = []
        for i in range(samples.shape[0]):
            sample = samples[i, ...]
            cv_sample = sample.permute(1, 2, 0).numpy()
            blurred_sample = torch.tensor(blur(cv_sample, (self.kernel_size, self.kernel_size)))
            if len(blurred_sample.shape) == 2:
                blurred_sample = blurred_sample.unsqueeze(-1)
            baseline = blurred_sample.permute(2, 0, 1)

            to_mask = torch.zeros(sample.shape).flatten(0 if self.feature_level == "channel" else 1)
            if self.feature_level == "channel":
                to_mask[indices[i]] = 1.
            else:
                to_mask[:, indices[i]] = 1.
            to_mask = to_mask.reshape(sample.shape)
            res.append(sample - (to_mask * sample) + (to_mask * baseline))
        return torch.stack(res, dim=0)
