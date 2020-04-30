from typing import Iterable, Callable, List, Dict
import numpy as np


# Returns a dictionary containing, for each given method, a list of Sensitivity-n values
# where the values of n are given by mask_range
def sensitivity_n(data: Iterable, model: Callable,
                  methods: Dict[str, Callable], mask_range: List[int],
                  n_subsets: int, mask_value: float):
    result = {m_name: [[] for _ in mask_range] for m_name in methods}
    for batch_index, (samples, labels) in enumerate(data):
        print(f"Batch {batch_index}...")
        sample_size = np.prod(samples.shape[1:])
        # Get original output and attributions
        orig_output = model(samples)
        attrs = {m_name: methods[m_name](samples, labels) for m_name in methods}
        for n_idx, n in enumerate(mask_range):
            output_diffs = []
            sum_of_attrs = {m_name: [] for m_name in methods}
            for _ in range(n_subsets):
                # Generate mask and masked samples
                # batch_dim: [batch_size, n] (made to match unravel_index output)
                mask = np.random.choice(sample_size, n)
                unraveled = np.unravel_index(mask, samples.shape[1:])
                batch_dim = np.array(list(range(samples.shape[0])) * n).reshape(-1, samples.shape[0]).transpose()
                masked_samples = samples.clone()
                masked_samples[(batch_dim, *unraveled)] = mask_value

                output = model(masked_samples)
                # Get difference in output confidence for desired class
                output_diffs.append((orig_output - output)[np.arange(samples.shape[0]), labels]
                                    .reshape(samples.shape[0], 1)
                                    .detach().numpy())
                # Get sum of attributions of masked pixels
                for m_name in methods:
                    sum_of_attrs[m_name].append(
                        attrs[m_name][(batch_dim, *unraveled)].detach().numpy()
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
