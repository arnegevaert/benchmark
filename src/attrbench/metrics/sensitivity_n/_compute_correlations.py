from typing import List, Dict

import numpy as np
import torch

from attrbench.lib.util import corrcoef

# TODO also handle just np array for attrs (for functional)
def _compute_correlations(attrs: np.ndarray, n_range: List[int],
                          output_diffs: Dict[str, Dict[int, np.ndarray]],
                          indices: Dict[int, np.ndarray]) -> Dict[str, torch.Tensor]:
    result = {afn: [] for afn in output_diffs.keys()}
    attrs = attrs.reshape((attrs.shape[0], 1, -1))  # [batch_size, 1, -1]
    for n in n_range:
        n_mask_attrs = np.take_along_axis(attrs, axis=-1, indices=indices[n])  # [batch_size, num_subsets, n]
        for fn in output_diffs.keys():
            # Calculate sums of attributions
            n_sum_of_attrs = n_mask_attrs.sum(axis=-1)  # [batch_size, num_subsets]
            n_output_diffs = output_diffs[fn][n]
            # Calculate correlation between output difference and sum of attribution values
            result[fn].append(corrcoef(n_sum_of_attrs, n_output_diffs))

    for afn in output_diffs.keys():
        result[afn] = torch.tensor(np.stack(result[afn], axis=1))
    return result
