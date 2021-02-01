from attrbench.lib.masking import Masker
import torch
import numpy as np


class RandomMasker(Masker):
    def __init__(self, feature_level, additive=False, std=1, num_samples=1):
        super().__init__(feature_level)
        self.additive = additive
        self.std = std
        self.num_samples = num_samples

    def mask(self, samples, indices):
        batch_size, num_channels, rows, cols = samples.shape
        num_indices = indices.shape[1]
        batch_dim = np.tile(range(batch_size), (num_indices, 1)).transpose()

        mean = samples if self.additive else torch.zeros(samples.shape, device=samples.device)
        baseline = torch.normal(mean=mean, std=self.std)

        to_mask = torch.zeros(samples.shape, device=samples.device).flatten(1 if self.feature_level == "channel" else 2)
        if self.feature_level == "channel":
            to_mask[batch_dim, indices] = 1.
        else:
            to_mask[batch_dim, :, indices] = 1.
        to_mask = to_mask.reshape(samples.shape)
        return samples - (to_mask * samples) + (to_mask * baseline)

    def predict_masked(self, samples, indices, model, return_masked_samples=False):
        preds = []
        masked = None
        for i in range(self.num_samples):
            masked = self.mask(samples, indices)
            with torch.no_grad():
                preds.append(model(masked).detach())
        preds = torch.stack(preds, dim=0)  # [num_samples, batch_size, num_outputs]
        preds = torch.mean(preds, dim=0)  # [batch_size, num_outputs]
        if return_masked_samples:
            return preds, masked
        return preds
