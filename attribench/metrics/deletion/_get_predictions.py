from typing import Callable, List, Dict

import torch

from attribench.activation_fns import ACTIVATION_FNS
from attribench.metrics.deletion._dataset import _MaskingDataset


def _get_predictions(
    masking_dataset: _MaskingDataset,
    labels: torch.Tensor,
    model: Callable,
    activation_fns: List[str],
    writer=None,
) -> Dict[str, torch.Tensor]:
    """

    :param masking_dataset:
    :param labels:
    :param model:
    :param activation_fns:
    :param writer:
    :return: Dictionary containing results for each activation function
    """
    preds = {fn: [] for fn in activation_fns}
    for i, batch in enumerate(masking_dataset):
        with torch.no_grad():
            predictions = model(batch)
        if writer is not None:
            writer.add_images("masked_samples", batch, global_step=i)
        for fn in activation_fns:
            preds[fn].append(
                ACTIVATION_FNS[fn](predictions)
                .gather(dim=1, index=labels.unsqueeze(-1))
                .cpu()
            )
    for afn in activation_fns:
        preds[afn] = torch.cat(
            preds[afn], dim=1
        ).cpu()  # [batch_size, len(mask_range)]
    return preds
