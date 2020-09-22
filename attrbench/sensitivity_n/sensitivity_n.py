from typing import Iterable, Callable, List, Dict, Union, Tuple
import numpy as np
from tqdm import tqdm
from attrbench.util import mask_pixels, sum_of_attributions
import torch


# TODO we now look at actual labels. Add option to look at model output instead
def sensitivity_n_batch(samples: torch.Tensor, labels: torch.Tensor, model: Callable,
                        attribution: Union[Callable, Dict[str, Callable]],
                        mask_range: Union[List[int], Tuple[int]], n_subsets: int, mask_value: float,
                        pixel_level_mask: bool, device: str):
    if isinstance(attribution, Dict):
        attrs = {m_name: attribution[m_name](samples, labels) for m_name in attribution}
        result = {m_name: {} for m_name in attribution}
    elif isinstance(attribution, Callable):
        attrs = attribution(samples, labels)
        result = {}
    else:
        raise TypeError(f"Attribution must be of type Dict or Callable, not {type(attribution)}")

    samples = samples.to(device)
    batch_size = samples.size(0)
    labels = labels.to(device)
    with torch.no_grad():
        orig_output = model(samples)

    for n in mask_range:
        output_diffs = []
        sum_of_attrs = {m_name: [] for m_name in attribution} if isinstance(attribution, Dict) else []
        for _ in range(n_subsets):
            # Generate mask and masked samples
            start_dim = 2 if pixel_level_mask else 1
            indices = np.random.choice(np.prod(samples.shape[start_dim:]), n*batch_size).reshape((batch_size, n))
            masked_samples = mask_pixels(samples, indices, mask_value, pixel_level_mask)
            # Get output on masked samples
            with torch.no_grad():
                output = model(masked_samples)
            # Get difference in output confidence for desired class
            output_diffs.append((orig_output - output).gather(dim=1, index=labels.unsqueeze(-1))
                                .cpu().detach().numpy())
            # Get sum of attribution values for masked inputs
            if isinstance(attribution, Dict):
                for m_name in attribution:
                    sum_of_attrs[m_name].append(sum_of_attributions(attrs[m_name], indices)
                                                .cpu().detach().numpy())
            else:
                sum_of_attrs.append(sum_of_attributions(attrs, indices))
        # Calculate correlation between output difference and sum of attribution values
        if isinstance(attribution, Dict):
            for m_name in attribution:
                sum_of_attrs[m_name] = np.hstack(sum_of_attrs[m_name])
                result[m_name][n] = [np.corrcoef(output_diffs[i], sum_of_attrs[m_name][i])[0, 1]
                                     for i in range(samples.size(0))]
        else:
            sum_of_attrs = np.hstack(sum_of_attrs)
            result[n] = [np.corrcoef(output_diffs[i], sum_of_attrs[i][0, 1]) for i in range(samples.size(0))]
    return result


def sensitivity_n(data: Iterable, model: Callable,
                  attribution: Union[Callable, Dict[str, Callable]],
                  mask_range: Union[List[int], Tuple[int]], n_subsets: int, mask_value: float,
                  pixel_level_mask: bool, device: str):
    if isinstance(attribution, Dict):
        result = {m_name: {n: [] for n in mask_range} for m_name in attribution}
    elif isinstance(attribution, Callable):
        result = {n: [] for n in mask_range}
    else:
        raise TypeError(f"Attribution must be of type Dict or Callable, not {type(attribution)}")

    for samples, labels in tqdm(data):
        # Get batch sensitivity-n
        batch_sens_n = sensitivity_n_batch(samples, labels, model, attribution, mask_range, n_subsets,
                                           mask_value, pixel_level_mask, device)
        # Merge in result
        if isinstance(attribution, Dict):
            for m_name in batch_sens_n:
                for n in mask_range:
                    result[m_name][n] += batch_sens_n[m_name][n]
        else:
            for n in mask_range:
                result[n] += batch_sens_n[n]

    # TODO result class that handles saving and loading data
    return result
