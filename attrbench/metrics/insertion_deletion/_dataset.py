import numpy as np
import torch

from attrbench.lib import AttributionWriter
from attrbench.lib import segment_samples
from attrbench.lib.masking import Masker


class _InsertionDeletionDataset:
    def __init__(self, mode: str, num_steps: int, samples: torch.tensor, attrs: np.ndarray, masker: Masker):
        if mode not in ["insertion", "deletion"]:
            raise ValueError("Mode must be insertion or deletion")
        self.mode = mode
        self.num_steps = num_steps
        self.samples = samples
        masker_constructor, masker_kwargs=masker
        self.masker = masker_constructor(samples, attrs,**masker_kwargs)
        total_features = self.masker.get_total_features()
        self.mask_range = list((np.linspace(0, 1, num_steps) * total_features)[1:-1].astype(np.int))

    def __len__(self):
        return len(self.mask_range)

    def __getitem__(self, item):
        num_to_mask = self.mask_range[item]
        masked_samples = self.masker.keep_top(num_to_mask) if self.mode == "insertion" else self.masker.mask_top(num_to_mask)
        return masked_samples


class _IrofIiofDataset:
    def __init__(self, mode: str, samples: torch.tensor, attrs: np.ndarray, masker: Masker, writer: AttributionWriter = None):
        self.mode = mode
        self.samples = samples
        self.sorted_indices = None
        # Override sorted_indices to use segment indices instead of pixel indices
        self.segmented_images = segment_samples(samples.cpu().numpy())
        # self.segmented_images = torch.tensor(segment_samples(samples.cpu().numpy()), device=samples.device)
        # Override masker
        masker_constructor, masker_kwargs=masker
        self.masker = masker_constructor(samples, attrs,**masker_kwargs, segmented_samples=self.segmented_images)
        if writer is not None:
            writer.add_images("segmented samples", self.segmented_images)

    # def set_attrs(self, attrs: np.ndarray):
    #     avg_attrs = segment_attributions(self.segmented_images, torch.tensor(attrs, device=self.samples.device))
    #     self.sorted_indices = avg_attrs.argsort()  # [batch_size, num_segments]

    def __len__(self):
        # Exclude fully masked/inserted image
        return self.masker.get_total_features() - 1

    def __getitem__(self, item):
        masked = self.masker.keep_top(item+1) if self.mode == "insertion" else self.masker.mask_top(item+1)
        return masked
