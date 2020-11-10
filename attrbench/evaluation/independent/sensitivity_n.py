from typing import Callable, List, Union, Tuple
import numpy as np
from attrbench.lib import sum_of_attributions, MaskingPolicy
import torch
import warnings


def sensitivity_n(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
                  n_range: Union[List[int], Tuple[int]], num_subsets: int, masking_policy: MaskingPolicy,
                  debug_mode=False):
    attrs = method(samples, labels)
    result = []
    batch_size = samples.size(0)

    with torch.no_grad():
        orig_output = model(samples)

    debug_data = []
    for n in n_range:
        output_diffs = []
        sum_of_attrs = []
        if debug_mode:
            debug_data.append({
                "indices": [],
                "masked_samples": [],
            })
        for _ in range(num_subsets):
            # Generate mask and masked samples
            # Mask is generated using replace=False, same mask is used for all samples in batch
            num_features = np.prod(attrs.shape[1:])
            indices = torch.tensor(np.random.choice(num_features, size=n, replace=False)).repeat(batch_size, 1)
            masked_samples = masking_policy(samples, indices)
            if debug_mode:
                debug_data[-1]["indices"].append(indices)
                debug_data[-1]["masked_samples"].append(masked_samples)
            # Get output on masked samples
            with torch.no_grad():
                output = model(masked_samples)
            # Get difference in output confidence for desired class
            output_diffs.append((orig_output - output).gather(dim=1, index=labels.unsqueeze(-1)))
            # Get sum of attribution values for masked inputs
            sum_of_attrs.append(sum_of_attributions(attrs, indices.to(attrs.device)))
        # [batch_size, num_subsets]
        sum_of_attrs = torch.cat(sum_of_attrs, dim=1)
        output_diffs = torch.cat(output_diffs, dim=1)
        # Calculate correlation between output difference and sum of attribution values
        # Subtract mean
        sum_of_attrs -= sum_of_attrs.mean(dim=1, keepdim=True)
        output_diffs -= output_diffs.mean(dim=1, keepdim=True)
        # Calculate covariances
        cov = (sum_of_attrs * output_diffs).sum(dim=1) / (num_subsets - 1)
        # Divide by product of standard deviations
        # [batch_size]
        denom = sum_of_attrs.std(dim=1)*output_diffs.std(dim=1)
        denom_zero = (denom == 0.)
        if torch.any(denom_zero):
            warnings.warn("Zero standard deviation detected.")
        corrcoefs = cov / (sum_of_attrs.std(dim=1)*output_diffs.std(dim=1))
        corrcoefs[denom_zero] = 0.
        result.append(corrcoefs)
    # [batch_size, len(n_range)]
    result = torch.stack(result, dim=1).cpu().detach()
    if debug_mode:
        debug_result = {
            "attrs": attrs,
            "pert_data": debug_data
        }
        return result, debug_result
    return result