import numpy as np
import torch

from attribench._segmentation import segment_samples
from attribench.masking import Masker
from attribench.masking.image import ImageMasker


class MaskingDataset:
    def __init__(self, mode: str, start: float, stop: float, num_steps: int):
        if mode not in ("morf", "lerf"):
            raise ValueError("Mode must be morf or lerf")
        if not ((0.0 <= start <= 1.0) and (0.0 <= stop <= 1.0)):
            raise ValueError("Start and stop must be between 0 and 1")
        self.mode = mode
        self.start = start
        self.stop = stop
        self.num_steps = num_steps

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


class DeletionDataset(MaskingDataset):
    def __init__(
        self,
        mode: str,
        start: float,
        stop: float,
        num_steps: int,
        samples: torch.Tensor,
        attrs: torch.Tensor,
        masker: Masker,
    ):
        super().__init__(mode, start, stop, num_steps)
        self.samples = samples
        self.masker = masker
        self.masker.set_batch(samples, attrs)

        total_features = self.masker.get_num_features()
        self.mask_range = list(
            (np.linspace(start, stop, num_steps) * total_features).astype(int)
        )

    def __len__(self):
        return len(self.mask_range)

    def __getitem__(self, item):
        if item >= len(self.mask_range):
            raise StopIteration
        num_to_mask = self.mask_range[item]
        masked_samples = (
            self.masker.mask_top(num_to_mask)
            if self.mode == "morf"
            else self.masker.mask_bot(num_to_mask)
        )
        return masked_samples


class IrofDataset(MaskingDataset):
    def __init__(
        self,
        mode: str,
        start: float,
        stop: float,
        num_steps: int,
        samples: torch.Tensor,
        masker: ImageMasker,
    ):
        super().__init__(mode, start, stop, num_steps)
        self.samples = samples
        self.masker = masker
        self.segmented_images = torch.tensor(
            segment_samples(samples.cpu().numpy()), device=samples.device
        )

    def __len__(self):
        return self.num_steps

    def __getitem__(self, item):
        if item >= self.num_steps:
            raise StopIteration
        to_mask = self.start + (item / (self.num_steps - 1)) * (
            self.stop - self.start
        )
        return (
            self.masker.mask_top(to_mask)
            if self.mode == "morf"
            else self.masker.mask_bot(to_mask)
        )

    def set_attrs(self, attrs: torch.Tensor):
        self.masker.set_batch(self.samples, attrs, self.segmented_images)
