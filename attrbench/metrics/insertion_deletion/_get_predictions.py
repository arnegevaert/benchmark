from typing import Callable, Tuple, Dict

import torch

from attrbench.lib.util import ACTIVATION_FNS


def _get_predictions(samples: torch.Tensor, labels: torch.Tensor,
                     model: Callable, ds,
                     activation_fns: Tuple[str], writer=None) -> Tuple[Dict, Dict, Dict]:
    with torch.no_grad():
        _orig_preds = model(samples)
        fully_masked = ds.masker.baseline
        _neutral_preds = model(fully_masked)
        binary_output = _orig_preds.shape[1] == 1
        if binary_output:
            # For deletion: measure decrease in output when removing features, but for class 0 in a binary classification
            # output will increase. So here we make outputs of class 0 positive so that these decrease when features removed
            original_preds_sign = torch.sign(_orig_preds)
            _orig_preds = _orig_preds * original_preds_sign
            _neutral_preds = _neutral_preds * original_preds_sign
        orig_preds = {fn: ACTIVATION_FNS[fn](_orig_preds).gather(dim=1, index=labels.unsqueeze(-1)).cpu()
                      for fn in activation_fns}

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
            predictions = predictions * original_preds_sign if binary_output else predictions
        if writer is not None:
            writer.add_images("masked_samples", batch, global_step=i)
        for fn in activation_fns:
            inter_preds[fn].append(ACTIVATION_FNS[fn](predictions).gather(dim=1, index=labels.unsqueeze(-1)).cpu())
    return orig_preds, neutral_preds, inter_preds

