import numpy as np
import torch

from attrbench.lib import AttributionWriter
from attrbench.lib import mask_segments, segment_samples, segment_attributions
from attrbench.lib.masking import Masker


class _DeletionDataset:
    def __init__(self, mode: str, start: float, stop: float, num_steps: int, samples: torch.tensor, attrs: np.ndarray, masker: Masker):
        if mode not in ("morf", "lerf"):
            raise ValueError("Mode must be morf or lerf")
        if not ((0. <= start <= 1.) and (0. <= stop <= 1.)):
            raise ValueError("Start and stop must be between 0 and 1")
        self.samples = samples
        self.masker = masker
        self.masker.initialize_baselines(samples)
        # Flatten each sample in order to sort indices per sample
        attrs = attrs.reshape(attrs.shape[0], -1)  # [batch_size, -1]
        # Sort indices of attrs in ascending order
        self.sorted_indices = np.argsort(attrs)
        if mode == "lerf":
            self.sorted_indices = self.sorted_indices[:, ::-1]

        total_features = attrs.shape[1]
        self.mask_range = list((np.linspace(start, stop, num_steps) * total_features).astype(np.int))

    def __len__(self):
        return len(self.mask_range)

    def __getitem__(self, item):
        num_to_mask = self.mask_range[item]
        if num_to_mask == 0:
            return self.samples.detach().clone()
        indices = self.sorted_indices[:, -num_to_mask:]
        masked_samples = self.masker.mask(self.samples, indices)
        return masked_samples


class _IrofIiofDataset:
    def __init__(self, mode: str, samples: torch.tensor, masker: Masker,
                 writer: AttributionWriter = None):
        self.mode = mode
        self.samples = samples
        self.masker = masker
        self.masker.initialize_baselines(samples)
        self.sorted_indices = None
        # Override sorted_indices to use segment indices instead of pixel indices
        self.segmented_images = torch.tensor(segment_samples(samples.cpu().numpy()), device=samples.device)
        if writer is not None:
            writer.add_images("segmented samples", self.segmented_images)

    def set_attrs(self, attrs: np.ndarray):
        avg_attrs = segment_attributions(self.segmented_images, torch.tensor(attrs, device=self.samples.device))
        self.sorted_indices = avg_attrs.argsort()  # [batch_size, num_segments]

    def __len__(self):
        # Exclude fully masked/inserted image
        return self.sorted_indices.shape[1] - 1

    def __getitem__(self, item):
        indices = self.sorted_indices[:, :-(item+1)] if self.mode == "insertion" else self.sorted_indices[:, -(item+1):]
        return mask_segments(self.samples, self.segmented_images, indices, self.masker)
