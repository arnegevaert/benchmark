from attrbench.lib.masking import Masker
from attrbench.lib import segment_attributions
from typing import List, Union, Optional
import numpy as np
import torch


class ImageMasker(Masker):
    def __init__(self, feature_level):
        if feature_level not in ("channel", "pixel"):
            raise ValueError(f"feature_level must be 'channel' or 'pixel'. Found {feature_level}.")
        self.feature_level = feature_level
        super().__init__()
        # will be set after initialize_batch:
        self.segmented_samples: Optional[torch.Tensor] = None
        self.segmented_attributions: Optional[np.ndarray] = None
        self.segment_indices: Optional[List[np.ndarray]] = None
        self.use_segments: bool = False

    def set_batch(self, samples: torch.Tensor, attributions: np.ndarray = None, segmented_samples: torch.Tensor = None):
        if attributions is not None and not self._check_attribution_shape(samples, attributions):
            raise ValueError(f"samples and attribution shape not compatible for feature level {self.feature_level}."
                             f"Found shapes {samples.shape} and {attributions.shape}")
        if segmented_samples is not None:
            if not (samples.shape[0] == segmented_samples.shape[0] and
                    samples.shape[-2:] == segmented_samples.shape[-2:]):
                raise ValueError(f"Incompatible shapes: {samples.shape}, {segmented_samples.shape}")
            if samples.device != segmented_samples.device:
                raise ValueError("Device for samples and segmented_samples must be equal."
                                 f"Got {samples.device} for samples, {segmented_samples.device} for segmented_samples.")

        self.samples = samples
        self.attributions = attributions
        self.segmented_samples = segmented_samples
        if segmented_samples is not None:
            self.segment_indices = [np.unique(segmented_samples.cpu().numpy()[i, ...]) for i in range(samples.shape[0])]
        self.use_segments = self.segmented_samples is not None

        if self.segmented_samples is not None and self.attributions is not None:
            self.segmented_attributions = segment_attributions(self.segmented_samples.cpu().numpy(), self.attributions)
            sorted_indices = self.segmented_attributions.argsort()

            # Filter out the -np.inf values from the sorted indices
            filtered_sorted_indices = []
            for i in range(self.segmented_samples.shape[0]):
                num_infs = np.count_nonzero(self.segmented_attributions[i, ...] == -np.inf)
                filtered_sorted_indices.append(sorted_indices[i, num_infs:])
            self.sorted_indices = filtered_sorted_indices

        elif self.attributions is not None:
            self.sorted_indices = self.attributions.reshape(self.attributions.shape[0], -1).argsort()

        self.initialize_baselines(self.samples)

    def get_num_features(self):
        if self.use_segments:
            raise ValueError("When using segments, total number of features varies per image.")
        if self.feature_level == "channel":
            return self.samples.flatten(1).shape[-1]
        if self.feature_level == "pixel":
            return self.samples.flatten(2).shape[-1]

    def mask_top(self, k):
        if not self.use_segments:
            return super().mask_top(k)
        if k == 0:
            return self.samples
        # When using segments, k is relative (between 0 and 1)
        indices = []
        for i in range(self.samples.shape[0]):
            num_segments = len(self.sorted_indices[i])
            num_to_mask = int(num_segments * k)
            indices.append(self.sorted_indices[i][-num_to_mask:])
        return self._mask_segments(indices)

    def mask_bot(self, k):
        if not self.use_segments:
            return super().mask_bot(k)
        if k == 0:
            return self.samples
        # When using segments, k is relative (between 0 and 1)
        indices = []
        for i in range(self.samples.shape[0]):
            num_segments = len(self.sorted_indices[i])
            num_to_mask = int(num_segments * k)
            indices.append(self.sorted_indices[i][:num_to_mask])
        return self._mask_segments(indices)

    def mask_rand(self, k, return_indices=False):
        if k == 0:
            return self.samples
        rng = np.random.default_rng()
        if not self.use_segments:
            return super().mask_rand(k, return_indices)
        else:
            # this is done a little different form super(),
            # no shuffle here: for each image, only select segments that exist in that image
            # k can be large -> rng.choice raises exception if k > number of segments
            indices = [rng.choice(self.segment_indices[i], size=k, replace=False)
                       for i in range(self.segmented_samples.shape[0])]
            masked_samples = self._mask_segments(indices)
        if return_indices:
            return masked_samples, indices
        return masked_samples

    def _check_attribution_shape(self, samples: torch.Tensor, attributions: np.ndarray):
        if self.feature_level == "channel":
            # Attributions should be same shape as samples
            return list(samples.shape) == list(attributions.shape)
        elif self.feature_level == "pixel":
            # attributions should have the same shape as samples,
            # except the channel dimension must be 1
            aggregated_shape = list(samples.shape)
            aggregated_shape[1] = 1
            return aggregated_shape == list(attributions.shape)

    def _mask(self, indices: np.ndarray):
        if self.baseline is None:
            raise ValueError("Masker was not initialized.")
        if self.use_segments:
            return self._mask_segments(indices)
        else:
            batch_size, num_channels, rows, cols = self.samples.shape
            num_indices = indices.shape[1]
            batch_dim = np.tile(range(batch_size), (num_indices, 1)).transpose()

            to_mask = torch.zeros(self.samples.shape, device=self.samples.device)\
                .flatten(1 if self.feature_level == "channel" else 2)
            #to_mask = np.zeros(self.samples.shape)
            if self.feature_level == "channel":
                to_mask[batch_dim, indices] = 1.
            else:
                try:
                    to_mask[batch_dim, :, indices] = 1.
                except IndexError:
                    raise ValueError("Masking index was out of bounds. "
                                     "Make sure the masking policy is compatible with method output.")
            return self._mask_boolean(to_mask.view(self.samples.shape).bool())

    def _mask_segments(self, segments: Union[np.ndarray, List[np.ndarray]]) -> torch.tensor:
        if not self.segmented_samples.shape[0] == len(segments):
            raise ValueError(
                f"Number of segment lists doesn't match number of images: {self.segmented_samples.shape[0]}"
                f"were expected, {len(segments)} were given.")
        bool_masks = []
        for i in range(self.samples.shape[0]):
            seg_img = self.segmented_samples[i, ...]
            segs = segments[i]
            bool_masks.append(_isin(seg_img, torch.tensor(segs, device=seg_img.device)))
        bool_masks = torch.stack(bool_masks, dim=0)
        if self.samples.shape[1] == 3:
            bool_masks = bool_masks.repeat(1, 3, 1, 1)
        return self._mask_boolean(bool_masks)

    def _mask_boolean(self, bool_mask: torch.tensor):
        return self.samples - (bool_mask * self.samples) + (bool_mask * self.baseline)

    def initialize_baselines(self, samples: torch.Tensor):
        raise NotImplementedError


def _isin(a: torch.Tensor, b: torch.Tensor):
    # https://stackoverflow.com/questions/60918304/get-indices-of-elements-in-tensor-a-that-are-present-in-tensor-b
    return (a[..., None] == b).any(-1)
