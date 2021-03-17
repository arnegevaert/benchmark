from attrbench.lib.masking import Masker
import torch


class RandomMasker(Masker):
    def __init__(self, feature_level, std=1, num_samples=1):
        super().__init__(feature_level)
        self.std = std
        self.num_samples = num_samples

    def initialize_baselines(self, samples: torch.tensor):
        self.baseline = torch.randn(*samples.shape, device=samples.device) * self.std
