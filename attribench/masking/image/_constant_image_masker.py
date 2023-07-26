from attribench.masking.image import ImageMasker
import torch


class ConstantImageMasker(ImageMasker):
    """Image masker that masks pixels or features by replacing them with
    a given constant value.
    """
    def __init__(self, masking_level: str, mask_value=0.0):
        """
        Parameters
        ----------
        feature_level : str
            The level at which to mask the image. Must be either
            ``"pixel"`` or ``"feature"``.
        mask_value : float
            The value to use for masking. Defaults to 0.0.
        """
        super().__init__(masking_level)
        self.mask_value = mask_value

    def _initialize_baselines(self, samples: torch.Tensor):
        self.baseline = (
            torch.ones(samples.shape, device=samples.device) * self.mask_value
        )
