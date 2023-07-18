from attribench.masking.image import ImageMasker
import torch


class SampleAverageImageMasker(ImageMasker):
    """Image masker that masks pixels or features by replacing them with
    the average value in the corresponding image.
    """
    def __init__(self, feature_level: str):
        """
        Parameters
        ----------
        feature_level : str
            The level at which to mask the image. Must be either
            ``"pixel"`` or ``"feature"``.
        """
        super().__init__(feature_level)

    def _initialize_baselines(self, samples: torch.Tensor):
        batch_size, num_channels, rows, cols = samples.shape
        self.baseline = (
            torch.mean(samples.flatten(2), dim=-1)
            .reshape(batch_size, num_channels, 1, 1)
            .repeat(1, 1, rows, cols)
        )
