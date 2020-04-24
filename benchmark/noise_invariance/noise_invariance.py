from benchmark.noise_invariance.noise_perturbed_dataset import NoisePerturbedDataset
from typing import Callable, Dict
import numpy as np


def noise_invariance(data: NoisePerturbedDataset, methods: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]):
    result = {}
    for m_name in methods:
        method = methods[m_name]
        print(f"Method: {m_name}...")
        diffs = [[] for _ in range(len(data.perturbation_levels))]
        cur_max_diff = 0
        cur_max_diff_examples = {}
        for batch_idx, batch in enumerate(data):
            orig = batch["original"]
            labels = batch["labels"]
            orig_attr = method(orig, labels)  # [batch_size, *sample_shape]
            for n_l, noise_level_batch in enumerate(batch["perturbed"]):
                perturbed_attr = method(noise_level_batch, labels)  # [batch_size, *sample_shape]
                avg_diff_per_image = np.average(np.reshape(np.abs(orig_attr - perturbed_attr), (data.batch_size, -1)),
                                                axis=1)  # [batch_size]
                diffs[n_l].append(avg_diff_per_image)
                max_diff_idx = np.argmax(avg_diff_per_image).item()
                if avg_diff_per_image[max_diff_idx] > cur_max_diff:
                    cur_max_diff = avg_diff_per_image[max_diff_idx]
                    cur_max_diff_examples = {
                        "orig": orig[max_diff_idx], "perturbed": noise_level_batch[max_diff_idx],
                        "orig_attr": orig_attr[max_diff_idx], "perturbed_attr": perturbed_attr[max_diff_idx],
                        "noise_level": data.perturbation_levels[n_l]
                    }
        diffs = [np.concatenate(n_l_diffs) for n_l_diffs in diffs]
        diffs = np.vstack(diffs).transpose()
        result[m_name] = {
            "diffs": diffs,  # [noise_levels, n_batches]
            "max_diff": cur_max_diff_examples,
            "max_diff_exs": cur_max_diff_examples
        }
    return result
