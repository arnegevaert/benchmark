import numpy as np
import torch

from attrbench.lib import AttributionWriter
from attrbench.lib import mask_segments, segment_samples, segment_attributions
from attrbench.lib.masking import Masker


class _InsertionDeletionDataset:
    def __init__(self, mode: str, num_steps: int, samples: torch.tensor, attrs: np.ndarray, masker: Masker,
                 reverse_order: bool = False):
        if mode not in ["insertion", "deletion"]:
            raise ValueError("Mode must be insertion or deletion")
        self.mode = mode
        self.num_steps = num_steps
        self.samples = samples
        self.masker = masker
        self.masker.initialize_baselines(samples)
        # Flatten each sample in order to sort indices per sample
        attrs = attrs.reshape(attrs.shape[0], -1)  # [batch_size, -1]
        # Sort indices of attrs in ascending order
        self.sorted_indices = np.argsort(attrs)
        if reverse_order:
            self.sorted_indices = np.flip(self.sorted_indices, axis=1)

        total_features = attrs.shape[1]
        self.mask_range = list((np.linspace(0, 1, num_steps) * total_features)[1:-1].astype(np.int))

    def __len__(self):
        return len(self.mask_range)

    def __getitem__(self, item):
        num_to_mask = self.mask_range[item]
        indices = self.sorted_indices[:, :-num_to_mask] if self.mode == "insertion" \
            else self.sorted_indices[:, -num_to_mask:]
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

    def set_attrs(self, attrs: np.ndarray, reverse_order: bool = False):
        avg_attrs = segment_attributions(self.segmented_images, torch.tensor(attrs, device=self.samples.device))
        self.sorted_indices = avg_attrs.argsort()  # [batch_size, num_segments]
        if reverse_order:
            self.sorted_indices = torch.flip(self.sorted_indices, dims=[1])

    def __len__(self):
        # Exclude fully masked/inserted image
        return self.sorted_indices.shape[1] - 1

    def __getitem__(self, item):
        indices = self.sorted_indices[:, :-(item+1)] if self.mode == "insertion" else self.sorted_indices[:, -(item+1):]
        return mask_segments(self.samples, self.segmented_images, indices, self.masker)
