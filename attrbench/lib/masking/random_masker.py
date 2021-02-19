from attrbench.lib.masking import Masker
import torch


class RandomMasker(Masker):
    def __init__(self, feature_level, std=1, num_samples=1):
        super().__init__(feature_level)
        self.std = std
        self.num_samples = num_samples

    def initialize_baselines(self, samples):
        mean = torch.zeros(samples.shape, device=samples.device)
        self.baseline = torch.normal(mean=mean, std=self.std)

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
