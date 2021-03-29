from typing import Tuple, Dict

import numpy as np
import torch

from attrbench.lib.util import corrcoef

_OUT_FNS = {
    "mse": lambda a, b: ((a - b) ** 2).mean(axis=1, keepdims=True),
    "corr": lambda a, b: corrcoef(a, b)[..., np.newaxis]
}


# TODO also handle just np array for attrs (for functional)
def _compute_result(pert_vectors: np.ndarray, pred_diffs: Dict[str, np.ndarray], attrs_dict: Dict[str, np.ndarray],
                    modes: Tuple[str]) -> Dict[str, Dict[str, torch.tensor]]:
    result = {}
    for key in attrs_dict.keys():
        attrs = attrs_dict[key]
        # Replicate attributions along channel dimension if necessary (if explanation has fewer channels than image)
        if attrs.shape[1] != pert_vectors.shape[-3]:
            attrs = np.repeat(attrs, pert_vectors.shape[-3], axis=1)

        # Calculate dot product between each sample and its corresponding perturbation vector
        # This is equivalent to diagonal of matmul
        attrs = attrs.reshape((attrs.shape[0], 1, -1))  # [batch_size, 1, -1]
        pert_vectors_flat = pert_vectors.reshape(
            (pert_vectors.shape[0], pert_vectors.shape[1], -1))  # [batch_size, num_perturbations, -1]
        dot_product = (attrs * pert_vectors_flat).sum(axis=-1)  # [batch_size, num_perturbations]

        result[key] = {}
        for mode in modes:
            result[key][mode] = {}
            for afn in pred_diffs.keys():
                result[key][mode][afn] = torch.tensor(_OUT_FNS[mode](dot_product, pred_diffs[afn]))
    return result
