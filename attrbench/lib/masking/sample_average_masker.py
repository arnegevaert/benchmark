from attrbench.lib.masking import ImageMasker
import torch


class SampleAverageMasker(ImageMasker):
    def __init__(self, feature_level):
        super().__init__(feature_level)

    def initialize_baselines(self, samples: torch.tensor):
        batch_size, num_channels, rows, cols = samples.shape
        self.baseline = torch.mean(samples.flatten(2), dim=-1) \
            .reshape(batch_size, num_channels, 1, 1) \
            .repeat(1, 1, rows, cols)
