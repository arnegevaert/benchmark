from typing import Tuple, Dict

import numpy as np
import torch

from attrbench.lib.util import corrcoef


def _normalized_mse(a: np.ndarray, b: np.ndarray):
    a = (a - np.mean(a, axis=1, keepdims=True))
    a_std = np.std(a, axis=1, keepdims=True)
    a = np.divide(a, a_std, out=a, where=a_std != 0)

    b = (b - np.mean(b, axis=1, keepdims=True))
    b_std = np.std(b, axis=1, keepdims=True)
    b = np.divide(b, b_std, out=b, where=b_std != 0)

    return ((a - b)**2).mean(axis=1, keepdims=True)


_LOSS_FNS = {
    "mse": lambda a, b: ((a - b) ** 2).mean(axis=1, keepdims=True),
    "normalized_mse": _normalized_mse,
    "corr": lambda a, b: corrcoef(a, b)[..., np.newaxis]
}


def _compute_result(pert_vectors: np.ndarray, pred_diffs: Dict[str, np.ndarray], attrs: np.ndarray,
                    loss_fns: Tuple[str]) -> Dict[str, Dict[str, torch.tensor]]:
    result = {}
    # Replicate attributions along channel dimension if necessary (if explanation has fewer channels than image)
    if attrs.shape[1] != pert_vectors.shape[-3]:
        attrs = np.repeat(attrs, pert_vectors.shape[-3], axis=1)

    # Calculate dot product between each sample and its corresponding perturbation vector
    # This is equivalent to diagonal of matmul
    attrs = attrs.reshape((attrs.shape[0], 1, -1))  # [batch_size, 1, -1]
    pert_vectors_flat = pert_vectors.reshape(
        (pert_vectors.shape[0], pert_vectors.shape[1], -1))  # [batch_size, num_perturbations, -1]

    attrs = torch.tensor(attrs).to("cuda")
    pert_vectors_flat = torch.tensor(pert_vectors_flat).to("cuda")
    # Calculate dot product in-place
    pert_vectors_flat *= attrs
    dot_product = pert_vectors_flat.sum(dim=-1).detach().cpu().numpy()  # [batch_size, num_perturbations]
    for afn in pred_diffs.keys():
        result[afn] = {}
        for loss_fn in loss_fns:
            result[afn][loss_fn] = torch.tensor(_LOSS_FNS[loss_fn](dot_product, pred_diffs[afn]))
    return result
