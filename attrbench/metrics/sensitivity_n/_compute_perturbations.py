from typing import Callable, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from attrbench.lib.util import ACTIVATION_FNS


def _compute_perturbations(samples: torch.Tensor, labels: torch.Tensor, ds: Dataset,
                           model: Callable, n_range, activation_fns: Tuple[str], writer=None) \
        -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[int, np.ndarray]]:
    dl = DataLoader(ds, shuffle=False, num_workers=4, pin_memory=True, batch_size=1)
    device = samples.device
    with torch.no_grad():
        orig_output = model(samples)

    output_diffs = {fn: {n: [] for n in n_range} for fn in activation_fns}
    removed_indices = {n: [] for n in n_range}
    for i, (batch, indices, n) in enumerate(dl):
        batch = batch[0].to(device).float()
        indices = indices[0].numpy()
        n = n.item()
        with torch.no_grad():
            output = model(batch)
        if writer is not None:
            writer.add_images(f"Masked samples N={n}", batch, global_step=i)
        for fn in activation_fns:
            fn_orig_out = ACTIVATION_FNS[fn](orig_output)
            fn_out = ACTIVATION_FNS[fn](output)
            output_diffs[fn][n].append((fn_orig_out - fn_out).gather(dim=1, index=labels.unsqueeze(-1)))  # [batch_size, 1]
        removed_indices[n].append(indices)  # [batch_size, n]

    for n in n_range:
        for fn in activation_fns:
            output_diffs[fn][n] = torch.cat(output_diffs[fn][n], dim=1).detach().cpu().numpy()  # [batch_size, num_subsets]
        removed_indices[n] = np.stack(removed_indices[n], axis=1)  # [batch_size, num_subsets, n]
    return output_diffs, removed_indices

