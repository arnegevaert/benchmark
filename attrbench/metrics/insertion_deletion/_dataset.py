import numpy as np
from torch.utils.data import Dataset

from attrbench.lib import AttributionWriter
from attrbench.lib import mask_segments, segment_samples_attributions
from attrbench.lib.masking import Masker


class _InsertionDeletionDataset(Dataset):
    def __init__(self, mode: str, num_steps: int, samples: np.ndarray, attrs: np.ndarray, masker: Masker,
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


class _IrofIiofDataset(_InsertionDeletionDataset):
    def __init__(self, mode: str, samples: np.ndarray, attrs: np.ndarray, masker: Masker,
                 reverse_order: bool = False, writer: AttributionWriter = None):
        super().__init__(mode, num_steps=100, samples=samples, attrs=attrs, masker=masker, reverse_order=reverse_order)
        # Override sorted_indices to use segment indices instead of pixel indices
        self.segmented_images, avg_attrs = segment_samples_attributions(samples, attrs)
        self.sorted_indices = avg_attrs.argsort()  # [batch_size, num_segments]
        if reverse_order:
            self.sorted_indices = np.flip(self.sorted_indices, axis=1)
        if writer is not None:
            writer.add_images("segmented samples", self.segmented_images)

    def __len__(self):
        # Exclude fully masked image
        return self.sorted_indices.shape[1] - 1

    def __getitem__(self, item):
        indices = self.sorted_indices[:, :-(item+1)] if self.mode == "insertion" else self.sorted_indices[:, -(item+1):]
        return mask_segments(self.samples, self.segmented_images, indices, self.masker)
