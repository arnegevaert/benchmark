from attrbench.lib.masking import Masker
import numpy as np
from cv2 import blur
import torch


class BlurringMasker(Masker):
    def __init__(self, feature_level, kernel_size):
        super().__init__(feature_level)
        self.kernel_size = kernel_size

    def mask(self, samples, indices):
        batch_size, num_channels, rows, cols = samples.shape
        num_indices = indices.shape[1]
        batch_dim = np.tile(range(batch_size), (num_indices, 1)).transpose()

        baseline = []
        for i in range(samples.shape[0]):
            sample = samples[i, ...]
            cv_sample = sample.permute(1, 2, 0).cpu().numpy()
            blurred_sample = torch.tensor(blur(cv_sample, (self.kernel_size, self.kernel_size)))
            if len(blurred_sample.shape) == 2:
                blurred_sample = blurred_sample.unsqueeze(-1)
            baseline.append(blurred_sample.permute(2, 0, 1).to(samples.device))
        baseline = torch.stack(baseline, dim=0)

        to_mask = torch.zeros(samples.shape, device=samples.device).flatten(1 if self.feature_level == "channel" else 2)
        if self.feature_level == "channel":
            to_mask[batch_dim, indices] = 1.
        else:
            to_mask[batch_dim, :, indices] = 1.
        to_mask = to_mask.reshape(samples.shape)
        return samples - (to_mask * samples) + (to_mask * baseline)
