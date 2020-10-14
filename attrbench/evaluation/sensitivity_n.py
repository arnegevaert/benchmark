from typing import Callable, List, Union, Tuple
import numpy as np
from attrbench.util import mask_pixels, sum_of_attributions
import torch
import warnings


# TODO we now look at actual labels. Add option to look at model output instead
def sensitivity_n(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
                  n_range: Union[List[int], Tuple[int]], num_subsets: int, mask_value: float):
    attrs = method(samples, labels)
    result = []
    batch_size = samples.size(0)

    with torch.no_grad():
        orig_output = model(samples)

    for n in n_range:
        output_diffs = []
        sum_of_attrs = []
        for _ in range(num_subsets):
            # Generate mask and masked samples
            indices = torch.tensor(np.random.choice(np.prod(attrs.shape[1:]), n*batch_size)).reshape((batch_size, n))
            masked_samples = mask_pixels(samples, indices, mask_value, pixel_level_mask=attrs.size(1) == 1)
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
    return torch.stack(result, dim=1).cpu().detach()
