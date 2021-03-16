from typing import Callable, List, Union, Tuple, Dict

import h5py
import numpy as np
import torch

from attrbench.lib import AttributionWriter
from attrbench.lib.util import corrcoef
from attrbench.metrics import Metric, MetricResult

_OUT_FNS = {
    "mse": lambda a, b: ((a - b) ** 2).mean(dim=1, keepdims=True),
    "corr": lambda a, b: torch.tensor(corrcoef(a.numpy(), b.numpy())).unsqueeze(-1)
}

def _compute_result(pert_vectors: torch.Tensor, pred_diffs: Dict[str, torch.Tensor], attrs: np.ndarray,
                    mode: Tuple[str]) -> Dict[str, Dict[str, torch.Tensor]]:
    # Replicate attributions along channel dimension if necessary (if explanation has fewer channels than image)
    attrs = torch.tensor(attrs).float()
    if attrs.shape[1] != pert_vectors.shape[-3]:
        shape = [1 for _ in range(len(attrs.shape))]
        shape[1] = pert_vectors.shape[-3]
        attrs = attrs.repeat(*tuple(shape))

    # Calculate dot product between each sample and its corresponding perturbation vector
    # This is equivalent to diagonal of matmul
    attrs = attrs.flatten(1).unsqueeze(1)  # [batch_size, 1, -1]
    pert_vectors = pert_vectors.flatten(2)  # [batch_size, num_perturbations, -1]
    dot_product = (attrs * pert_vectors).sum(dim=-1)  # [batch_size, num_perturbations]

    result = {}
    for mode in mode:
        result[mode] = {}
        for afn in pred_diffs.keys():
            result[mode][afn] = _OUT_FNS[mode](dot_product, pred_diffs[afn])
    return result

