from attrbench.lib.masking import ImageMasker
import torch


class RandomMasker(ImageMasker):
    def __init__(self, samples, attributions, feature_level, std=1, num_samples=1, segmented_samples =None):
        super().__init__(samples, attributions, feature_level, segmented_samples)
        self.std = std
        self.num_samples = num_samples
        self.initialize_baselines(samples)

    def initialize_baselines(self, samples: torch.tensor):
        self.baseline = torch.randn(*samples.shape, device=samples.device) * self.std
