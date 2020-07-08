from typing import Iterable, Callable, List, Dict
import numpy as np
from tqdm import tqdm
import torch


# Returns a dictionary containing, for each given method, a list of Sensitivity-n values
# where the values of n are given by mask_range
def sensitivity_n(data: Iterable, model: Callable,
                  methods: Dict[str, Callable], mask_range: List[int],
                  n_subsets: int, mask_value: float, pixel_level_mask: bool,
                  device: str):
    result = {m_name: [[] for _ in mask_range] for m_name in methods}
    for batch_index, (samples, labels) in enumerate(tqdm(data)):
        samples = samples.to(device)
        labels = labels.to(device)
        sample_size = np.prod(samples.shape[2:]) if pixel_level_mask else np.prod(samples.shape[1:])
        # Get original output and attributions
        orig_output = model(samples)
        attrs = {m_name: methods[m_name](samples, labels) for m_name in methods}
        for n_idx, n in enumerate(mask_range):
            output_diffs = []
            sum_of_attrs = {m_name: [] for m_name in methods}
            for _ in range(n_subsets):
                # Generate mask and masked samples
                masked_samples = samples.clone()
                mask = np.random.choice(sample_size, n)
                if pixel_level_mask:
                    # Mask on pixel level (aggregate across channels)
                    unraveled = np.unravel_index(mask, samples.shape[2:])
                    masked_samples[:, :, unraveled[0], unraveled[1]] = mask_value
                else:
                    # Mask on feature level (separate channels)
                    unraveled = np.unravel_index(mask, samples.shape[1:])
                    masked_samples[:, unraveled[0], unraveled[1], unraveled[2]] = mask_value

                output = model(masked_samples)
                # Get difference in output confidence for desired class
                output_diffs.append((orig_output - output)[np.arange(samples.shape[0]), labels]
                                    .reshape(samples.shape[0], 1)
                                    .cpu().detach().numpy())
                # Get sum of attributions of masked pixels
                for m_name in methods:
                    if pixel_level_mask:
                        mask_attrs = attrs[m_name][:, unraveled[0], unraveled[1]]
                    else:
                        mask_attrs = attrs[m_name][:, unraveled[0], unraveled[1], unraveled[2]]
                    sum_of_attrs[m_name].append(
                        mask_attrs.cpu().detach().numpy()
                        .reshape(samples.shape[0], -1)
                        .sum(axis=1)
                        .reshape(samples.shape[0], 1))
            output_diffs = np.hstack(output_diffs)
            for m_name in methods:
                sum_of_attrs[m_name] = np.hstack(sum_of_attrs[m_name])
                result[m_name][n_idx] += [np.corrcoef(output_diffs[i], sum_of_attrs[m_name][i])[0, 1]
                                          for i in range(samples.shape[0])]
    for m_name in methods:
        result[m_name] = np.array(result[m_name]).transpose()
    return result
