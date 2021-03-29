from typing import Callable, Tuple, Dict

import torch

from attrbench.lib.util import ACTIVATION_FNS


def _get_predictions(samples: torch.Tensor, labels: torch.Tensor,
                     model: Callable, ds,
                     activation_fns: Tuple[str], writer=None) -> Tuple[Dict, Dict, Dict]:
    with torch.no_grad():
        _orig_preds = model(samples)
        orig_preds = {fn: ACTIVATION_FNS[fn](_orig_preds).gather(dim=1, index=labels.unsqueeze(-1)).cpu()
                      for fn in activation_fns}
        fully_masked = ds.masker.baseline
        _neutral_preds = model(fully_masked)
        neutral_preds = {fn: ACTIVATION_FNS[fn](_neutral_preds).gather(dim=1, index=labels.unsqueeze(-1)).cpu()
                         for fn in activation_fns}
        if writer is not None:
            writer.add_images("orig_samples", samples)
            writer.add_images("neutral_samples", fully_masked)

    inter_preds = {fn: [] for fn in activation_fns}
    for i in range(len(ds)):
        batch = ds[i]
        with torch.no_grad():
            predictions = model(batch)
        if writer is not None:
            writer.add_images("masked_samples", batch, global_step=i)
        for fn in activation_fns:
            inter_preds[fn].append(ACTIVATION_FNS[fn](predictions).gather(dim=1, index=labels.unsqueeze(-1)).cpu())
    return orig_preds, neutral_preds, inter_preds

