import numpy as np
from torch.utils.data import Dataset

from attrbench.lib import AttributionWriter
from attrbench.lib import segment_samples
from attrbench.lib.masking import Masker


class _InsertionDeletionDataset(Dataset):
    def __init__(self, mode: str, num_steps: int, samples: np.ndarray, attrs: np.ndarray, masker: Masker,
                 reverse_order: bool = False):
        if mode not in ["insertion", "deletion"]:
            raise ValueError("Mode must be insertion or deletion")
        self.mode = mode
        self.num_steps = num_steps
        self.samples = samples
        masker_constructor, masker_kwargs=masker
        self.masker = masker_constructor(samples, attrs,**masker_kwargs)
        self.reverse_order = reverse_order
        total_features = self.masker.get_total_features()
        self.mask_range = list((np.linspace(0, 1, num_steps) * total_features)[1:-1].astype(np.int))

    def __len__(self):
        return len(self.mask_range)

    def __getitem__(self, item):
        num_to_mask = self.mask_range[item]
        if not self.reverse_order:
            masked_samples = self.masker.keep_top(num_to_mask) if self.mode == "insertion" else self.masker.mask_top(num_to_mask)
        else:
            masked_samples = self.masker.keep_bot(num_to_mask) if self.mode == "insertion" else self.masker.mask_bot(
                num_to_mask)
        return masked_samples


class _IrofIiofDataset(_InsertionDeletionDataset):
    def __init__(self, mode: str, samples: np.ndarray, attrs: np.ndarray, masker: Masker,
                 reverse_order: bool = False, writer: AttributionWriter = None):
        super().__init__(mode, num_steps=100, samples=samples, attrs=attrs, masker=masker, reverse_order=reverse_order)
        # Override sorted_indices to use segment indices instead of pixel indices
        self.segmented_images = segment_samples(samples)
        # Override masker
        masker_constructor, masker_kwargs=masker
        self.masker = masker_constructor(samples, attrs,**masker_kwargs, segmented_samples=self.segmented_images)
        if writer is not None:
            writer.add_images("segmented samples", self.segmented_images)

    def __len__(self):
        # Exclude fully masked/inserted image
        return self.masker.get_total_features() - 1

    def __getitem__(self, item):
        if not self.reverse_order:
            masked = self.masker.keep_top(item+1) if self.mode == "insertion" else self.masker.mask_top(item+1)
        else:
            masked = self.masker.keep_bot(item+1) if self.mode == "insertion" else self.masker.mask_bot(item+1)
        return masked
