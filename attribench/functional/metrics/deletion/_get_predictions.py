from typing import Callable, List, Dict

import torch

from attribench._activation_fns import ACTIVATION_FNS
from ._dataset import MaskingDataset


def get_predictions(
    masking_dataset: MaskingDataset,
    labels: torch.Tensor,
    model: Callable,
    activation_fns: List[str],
) -> Dict[str, torch.Tensor]:
    preds = {fn: [] for fn in activation_fns}
    for i, batch in enumerate(iter(masking_dataset)):
        with torch.no_grad():
            predictions = model(batch)
        for fn in activation_fns:
            preds[fn].append(
                ACTIVATION_FNS[fn](predictions)
                .gather(dim=1, index=labels.unsqueeze(-1))
                .cpu()
            )
    preds_cat = {}
    for afn in activation_fns:
        preds_cat[afn] = torch.cat(
            preds[afn], dim=1
        ).cpu()  # [batch_size, len(mask_range)]
    return preds_cat
