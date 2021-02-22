import torch
import numpy as np
from attrbench.lib.masking import Masker
from skimage.segmentation import slic
from typing import Callable
from attrbench.functional.seg.util import mask_segments, segment_samples_attributions


def seg_sensitivity_n(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: torch.Tensor,
                      min_subset_size: float, max_subset_size: float, num_steps: int, num_subsets: int,
                      masker: Masker, writer=None):
    # Segment images and attributions
    segmented_images, avg_attrs = segment_samples_attributions(samples.detach().cpu().numpy(),
                                                               attrs.detach().cpu().numpy())

    # Initialize masker
    masker.initialize_baselines(samples)