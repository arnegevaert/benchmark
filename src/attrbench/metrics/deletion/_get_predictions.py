from typing import Callable, List, Dict

import torch

from attrbench.lib.util import ACTIVATION_FNS
from attrbench.metrics.deletion._dataset import _MaskingDataset


def _get_predictions(masking_dataset: _MaskingDataset, labels: torch.Tensor,
                     model: Callable,
                     activation_fns: List[str], writer=None) -> Dict:
    preds = {fn: [] for fn in activation_fns}
    for i, batch in enumerate(masking_dataset):
        with torch.no_grad():
            predictions = model(batch)
        if writer is not None:
            writer.add_images("masked_samples", batch, global_step=i)
        for fn in activation_fns:
            preds[fn].append(ACTIVATION_FNS[fn](predictions).gather(dim=1, index=labels.unsqueeze(-1)).cpu())
    for afn in activation_fns:
        preds[afn] = torch.cat(preds[afn], dim=1).cpu()  # [batch_size, len(mask_range)]
    return preds
