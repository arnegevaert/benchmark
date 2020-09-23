from typing import Callable, List, Dict, Union, Tuple
import numpy as np
from attrbench.util import mask_pixels, sum_of_attributions
import torch


# TODO we now look at actual labels. Add option to look at model output instead
def sensitivity_n(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
                  n_range: Union[List[int], Tuple[int]], num_subsets: int, mask_value: float,
                  pixel_level_mask: bool, device: str):
    attrs = method(samples, labels)
    result = []
    samples = samples.to(device)
    batch_size = samples.size(0)
    start_dim = 2 if pixel_level_mask else 1
    sample_size = np.prod(samples.shape[start_dim:])
    labels = labels.to(device)

    with torch.no_grad():
        orig_output = model(samples)

    for n in n_range:
        output_diffs = []
        sum_of_attrs = []
        for _ in range(num_subsets):
            # Generate mask and masked samples
            indices = torch.tensor(np.random.choice(sample_size, n*batch_size)).reshape((batch_size, n))
            masked_samples = mask_pixels(samples, indices, mask_value, pixel_level_mask)
            # Get output on masked samples
            with torch.no_grad():
                output = model(masked_samples)
            # Get difference in output confidence for desired class
            output_diffs.append((orig_output - output).gather(dim=1, index=labels.unsqueeze(-1))
                                .cpu().detach().numpy())
            # Get sum of attribution values for masked inputs
            sum_of_attrs.append(sum_of_attributions(attrs, indices))
        # Calculate correlation between output difference and sum of attribution values
        sum_of_attrs = np.hstack(sum_of_attrs)
        output_diffs = np.hstack(output_diffs)
        result.append([np.corrcoef(output_diffs[i], sum_of_attrs[i])[0, 1] for i in range(samples.size(0))])
    # [batch_size, len(n_range)]
    return torch.tensor(result).t()