from attrbench.lib.masking import Masker
import torch


class RandomMasker(Masker):
    def __init__(self, feature_level, additive=False, std=1, num_samples=10):
        super().__init__(feature_level)
        self.additive = additive
        self.std = std
        self.num_samples = num_samples

    def mask(self, samples, indices):
        res = []
        for i in range(samples.shape[0]):
            sample = samples[i, ...]
            mean = sample if self.additive else torch.zeros(sample.shape, device=samples.device)
            baseline = torch.normal(mean=mean, std=self.std, device=samples.device)
            to_mask = torch.zeros(sample.shape, device=samples.device).flatten(0 if self.feature_level == "channel" else 1)
            if self.feature_level == "channel":
                to_mask[indices[i]] = 1.
            else:
                to_mask[:, indices[i]] = 1.
            to_mask = to_mask.reshape(sample.shape)
            res.append(sample - (to_mask * sample) + (to_mask * baseline))
        return torch.stack(res, dim=0)

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
