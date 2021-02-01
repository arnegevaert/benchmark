from attrbench.lib.masking import Masker
import torch
import numpy as np


class SampleAverageMasker(Masker):
    def mask(self, samples, indices):
        batch_size, num_channels, rows, cols = samples.shape
        num_indices = indices.shape[1]
        batch_dim = np.tile(range(batch_size), (num_indices, 1)).transpose()
        baseline = torch.mean(samples.flatten(2), dim=-1).reshape(batch_size, num_channels, 1, 1).repeat(1, 1, rows, cols)
        to_mask = torch.zeros(samples.shape, device=samples.device).flatten(1 if self.feature_level == "channel" else 2)
        if self.feature_level == "channel":
            to_mask[batch_dim, indices] = 1.
        else:
            to_mask[batch_dim, :, indices] = 1.
        to_mask = to_mask.reshape(samples.shape)
        return samples - (to_mask * samples) + (to_mask * baseline)
