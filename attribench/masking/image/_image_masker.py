from attribench.masking import Masker
from attribench._segmentation import segment_attributions
from typing import List, Union, Optional, Tuple
import numpy as np
import torch
from abc import abstractmethod


class ImageMasker(Masker):
    """Abstract base class for all image maskers.

    Image maskers are specific maskers for image data. They can be used to
    mask features on the feature level or the pixel level. If the masker is
    used on the feature level, the attributions must have the same shape as
    the samples. If the masker is used on the pixel level, the attributions
    must have the same shape as the samples, except the channel dimension
    must be 1.

    If the masker is used on the feature level, masking a feature means
    masking a specific input feature for all samples (i.e. one color value for
    one pixel). If the masker is used on the pixel level, masking a feature
    means masking a specific pixel for all samples (i.e. all color values for
    one pixel).

    If segmented samples are provided, the masker will use the segments to
    mask features. This means that masking a feature will mask all pixels
    belonging to the same segment. The attribution value of a segment is
    defined as the average attribution value of all features in that segment.
    """

    def __init__(self, masking_level: str):
        """
        Parameters
        ----------
        masking_level : str
            Either ``"feature"`` or ``"pixel"``. If ``"feature"``, the masker
            will mask features. If ``"pixel"``, the masker will mask pixels.

        Raises
        ------
        ValueError
            If ``masking_level`` is not ``"feature"`` or ``"pixel"``.
        """
        if masking_level not in ("feature", "pixel"):
            raise ValueError(
                f"feature_level must be 'feature' or 'pixel'."
                f" Found {masking_level}."
            )
        self.masking_level = masking_level
        super().__init__()
        # will be set after initialize_batch:
        self.segmented_samples: Optional[torch.Tensor] = None
        self.segmented_attributions: Optional[np.ndarray] = None
        self.segment_indices: Optional[List[np.ndarray]] = None
        self.use_segments: bool = False
        self.sorted_indices: torch.Tensor | List[torch.Tensor] | None = None

    def set_batch(
        self,
        samples: torch.Tensor,
        attributions: torch.Tensor | None = None,
        segmented_samples: torch.Tensor | None = None,
    ):
        """Set the batch of samples and attributions to use for masking.
        Optionally also set the segmented samples.

        Parameters
        ----------
        samples : torch.Tensor
            Samples to use for masking.
        attributions : torch.Tensor, optional
            Attributions to use for masking, by default None
            If None, the :meth:`mask_top` and :meth:`mask_bot` methods will
            not be available.
        segmented_samples : torch.Tensor, optional
            Segmented samples to use for masking, by default None
        """

        # Check if attributions are compatible with samples
        if attributions is not None and not self._check_attribution_shape(
            samples, attributions
        ):
            raise ValueError(
                f"samples and attribution shape not compatible for masking "
                f"level {self.masking_level}."
                f"Found shapes {samples.shape} and {attributions.shape}"
            )

        # Check if segmented samples are compatible with samples
        if segmented_samples is not None:
            if not (
                samples.shape[0] == segmented_samples.shape[0]
                and samples.shape[-2:] == segmented_samples.shape[-2:]
            ):
                raise ValueError(
                    f"Incompatible shapes: {samples.shape}, {segmented_samples.shape}"
                )
            if samples.device != segmented_samples.device:
                raise ValueError(
                    "Device for samples and segmented_samples must be equal."
                    f"Got {samples.device} for samples,"
                    f" {segmented_samples.device} for segmented_samples."
                )

        # Set attributes
        self.samples = samples
        self.attributions = attributions
        self.segmented_samples = segmented_samples
        self.use_segments = segmented_samples is not None

        if segmented_samples is not None and attributions is not None:
            # If segmented samples and attributions are given,
            # segment the attributions accordingly
            # and sort the segments
            self.segmented_attributions = segment_attributions(
                segmented_samples.cpu().numpy(),
                attributions.cpu().numpy(),
            )
            sorted_indices = torch.tensor(
                self.segmented_attributions.argsort()
            )

            # Filter out the -np.inf values from the sorted indices
            filtered_sorted_indices = []
            for i in range(segmented_samples.shape[0]):
                num_infs = np.count_nonzero(
                    self.segmented_attributions[i, ...] == -np.inf
                )
                filtered_sorted_indices.append(sorted_indices[i, num_infs:])
            self.sorted_indices = filtered_sorted_indices
        elif attributions is not None:
            # If only attributions are given, sort them
            self.sorted_indices = torch.tensor(
                attributions.cpu()
                .numpy()
                .reshape(attributions.shape[0], -1)
                .argsort()
            )

        if segmented_samples is not None:
            # Get the indices of the segments for each image
            self.segment_indices = [
                np.unique(segmented_samples.cpu().numpy()[i, ...])
                for i in range(samples.shape[0])
            ]

        self._initialize_baselines(self.samples)

    def get_num_features(self):
        assert self.samples is not None
        if self.use_segments:
            raise ValueError(
                "When using segments, total number of features varies per image."
            )
        if self.masking_level == "feature":
            return self.samples.flatten(1).shape[-1]
        if self.masking_level == "pixel":
            return self.samples.flatten(2).shape[-1]

    def mask_top(self, k: int):
        assert self.samples is not None
        assert self.sorted_indices is not None
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

    def mask_bot(self, k: int):
        assert self.samples is not None
        assert self.sorted_indices is not None
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

    def mask_rand(
        self, k: int, return_indices=False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        assert self.samples is not None
        if k == 0:
            return self.samples
        rng = np.random.default_rng()
        if not self.use_segments:
            return super().mask_rand(k, return_indices)
        else:
            # this is done a little different form super(),
            # no shuffle here: for each image, only select segments that exist in that image
            # k can be large -> rng.choice raises exception if k > number of segments
            assert self.segment_indices is not None
            assert self.segmented_samples is not None
            indices = torch.stack(
                [
                    torch.tensor(
                        rng.choice(
                            self.segment_indices[i], size=k, replace=False
                        )
                    )
                    for i in range(self.segmented_samples.shape[0])
                ]
            )
            masked_samples = self._mask_segments(indices)
        if return_indices:
            return masked_samples, indices
        return masked_samples

    def _check_attribution_shape(
        self, samples: torch.Tensor, attributions: torch.Tensor
    ):
        if self.masking_level == "feature":
            # Attributions should be same shape as samples
            return list(samples.shape) == list(attributions.shape)
        elif self.masking_level == "pixel":
            # attributions should have the same shape as samples,
            # except the channel dimension must be 1
            aggregated_shape = list(samples.shape)
            aggregated_shape[1] = 1
            return aggregated_shape == list(attributions.shape)

    def _mask(self, indices: torch.Tensor) -> torch.Tensor:
        if self.baseline is None:
            raise ValueError("Masker was not initialized.")
        if self.use_segments:
            return self._mask_segments(indices)
        else:
            assert self.samples is not None
            batch_size, _, _, _ = self.samples.shape
            num_indices = indices.shape[1]
            batch_dim = np.tile(
                range(batch_size), (num_indices, 1)
            ).transpose()

            to_mask = torch.zeros(
                self.samples.shape, device=self.samples.device
            ).flatten(1 if self.masking_level == "feature" else 2)
            # to_mask = np.zeros(self.samples.shape)
            if self.masking_level == "feature":
                to_mask[batch_dim, indices] = 1.0
            else:
                try:
                    to_mask[batch_dim, :, indices] = 1.0
                except IndexError:
                    raise ValueError(
                        "Masking index was out of bounds. "
                        "Make sure the masking policy is compatible with method output."
                    )
            return self._mask_boolean(to_mask.view(self.samples.shape).bool())

    def _mask_segments(
        self, segments: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        assert self.segmented_samples is not None
        assert self.samples is not None
        if not self.segmented_samples.shape[0] == len(segments):
            raise ValueError(
                f"Number of segment lists doesn't match number of images:"
                f" {self.segmented_samples.shape[0]}"
                f"were expected, {len(segments)} were given."
            )
        bool_masks = []
        for i in range(self.samples.shape[0]):
            seg_img = self.segmented_samples[i, ...]
            segs = segments[i].to(seg_img.device)
            bool_masks.append(_isin(seg_img, segs))
        bool_masks = torch.stack(bool_masks, dim=0)
        if self.samples.shape[1] == 3:
            bool_masks = bool_masks.repeat(1, 3, 1, 1)
        return self._mask_boolean(bool_masks)

    def _mask_boolean(self, bool_mask: torch.Tensor) -> torch.Tensor:
        return (
            self.samples
            - (bool_mask * self.samples)
            + (bool_mask * self.baseline)
        )

    @abstractmethod
    def _initialize_baselines(self, samples: torch.Tensor):
        raise NotImplementedError


def _isin(a: torch.Tensor, b: torch.Tensor):
    # https://stackoverflow.com/questions/60918304/get-indices-of-elements-in-tensor-a-that-are-present-in-tensor-b
    return (a[..., None] == b).any(-1)
