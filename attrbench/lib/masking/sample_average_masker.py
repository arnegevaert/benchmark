from attrbench.lib.masking import ImageMasker
import torch


class SampleAverageMasker(ImageMasker):
    def __init__(self, samples, attributions, feature_level, segmented_samples=None):
        super().__init__(samples, attributions, feature_level, segmented_samples)
        self.initialize_baselines(samples)
    def initialize_baselines(self, samples):
        batch_size, num_channels, rows, cols = samples.shape
        self.baseline = torch.mean(samples.flatten(2), dim=-1)\
            .reshape(batch_size, num_channels, 1, 1)\
            .repeat(1, 1, rows, cols)
