from typing import Dict

import numpy as np
import torch


def _mse(a, b):
    return ((a - b) ** 2).mean(axis=1, keepdims=True)


def _compute_result(pert_vectors: np.ndarray, pred_diffs: Dict[str, np.ndarray],
                    attrs: np.ndarray) -> Dict[str, torch.tensor]:
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
        result[afn] = torch.tensor(_mse(dot_product, pred_diffs[afn]))
    return result
