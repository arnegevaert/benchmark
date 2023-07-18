from attribench.masking.image import ImageMasker
import torch


class RandomImageMasker(ImageMasker):
    """Image masker that masks images with normally distributed random
    noise, with a given standard deviation.
    """
    def __init__(self, masking_level: str, std=1):
        """
        Parameters
        ----------
        masking_level : str
            The level at which to mask the image. Must be either
            ``"pixel"`` or ``"feature"``.
        std : float
            The standard deviation of the random noise to add to the image.
            Defaults to 1.
        """
        super().__init__(masking_level)
        self.std = std

    def _initialize_baselines(self, samples: torch.Tensor):
        self.baseline = (
            torch.randn(*samples.shape, device=samples.device) * self.std
        )
