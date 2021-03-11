from typing import Callable, Tuple, Dict

import torch
from torch.utils.data import DataLoader

from attrbench.lib.util import ACTIVATION_FNS
from ._dataset import _InsertionDeletionDataset


def _get_predictions(samples: torch.Tensor, labels: torch.Tensor,
                     model: Callable, ds: _InsertionDeletionDataset,
                     activation_fns: Tuple[str], writer=None) -> Tuple[Dict, Dict, Dict]:
    device = samples.device
    with torch.no_grad():
        _orig_preds = model(samples)
        orig_preds = {fn: ACTIVATION_FNS[fn](_orig_preds).gather(dim=1, index=labels.unsqueeze(-1))
                      for fn in activation_fns}
        fully_masked = torch.tensor(ds.masker.baseline, device=device, dtype=torch.float)
        _neutral_preds = model(fully_masked.to(device))
        neutral_preds = {fn: ACTIVATION_FNS[fn](_neutral_preds).gather(dim=1, index=labels.unsqueeze(-1))
                         for fn in activation_fns}
    dl = DataLoader(ds, shuffle=False, num_workers=4, pin_memory=True, batch_size=1)

    inter_preds = {fn: [] for fn in activation_fns}
    for i, batch in enumerate(dl):
        batch = batch[0].to(device).float()
        with torch.no_grad():
            predictions = model(batch)
        if writer is not None:
            writer.add_images("masked_samples", batch, global_step=i)
        for fn in activation_fns:
            inter_preds[fn].append(ACTIVATION_FNS[fn](predictions).gather(dim=1, index=labels.unsqueeze(-1)))
    return orig_preds, neutral_preds, inter_preds

