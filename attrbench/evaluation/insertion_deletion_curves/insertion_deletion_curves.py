from typing import Iterable, Callable, List, Dict
import numpy as np
from tqdm import tqdm
from attrbench.evaluation.result import LinePlotResult
from attrbench.evaluation.util import transform_fns, mask_pixels, insert_pixels
import torch


_MASK_METHODS = {
    "deletion": mask_pixels,
    "insertion": insert_pixels
}


def insertion_deletion_curves(data: Iterable, sample_shape, model: Callable, methods: Dict[str, Callable],
                              mask_range: List[int], mask_value: float, pixel_level_mask: bool,
                              device: str, mode: str, output_transform: str):
    assert output_transform in ["identity", "softmax", "logit_softmax"]
    assert mode in ["deletion", "insertion"]
    result = {m_name: [] for m_name in methods}
    for batch_index, (samples, labels) in enumerate(tqdm(data)):
        samples = samples.to(device)
        labels = labels.to(device)
        # Check which samples are classified correctly
        # Only want to calculate for correctly classified samples
        with torch.no_grad():
            y_pred = torch.argmax(model(samples), dim=1)
        samples = samples[y_pred == labels]
        labels = labels[y_pred == labels]
        batch_size = samples.size(0)
        if batch_size > 0:
            for key in methods:
                batch_result = []
                attrs = methods[key](samples, labels)  # [batch_size, *sample_shape]
                # Flatten each sample in order to sort indices per sample
                attrs = attrs.flatten(1)  # [batch_size, -1]
                # Sort indices of attrs in ascending order
                sorted_indices = attrs.argsort().cpu().detach().numpy()

                for i in mask_range:
                    # Mask/insert pixels
                    if i > 0:
                        masked_samples = _MASK_METHODS[mode](samples, sorted_indices[:, -i:], mask_value, pixel_level_mask)
                    else:
                        masked_samples = samples if mode == "deletion" else torch.ones(samples.shape).to(device) * mask_value
                    # Get predictions for result
                    with torch.no_grad():
                        predictions = transform_fns[output_transform](model(masked_samples))\
                            .gather(dim=1, index=labels.unsqueeze(-1))
                    batch_result.append(predictions.cpu().detach().numpy())
                batch_result = np.concatenate(batch_result, axis=1)
                result[key].append(batch_result)
    res_data = {
        # [n_batches*batch_size, len(mask_range)]
        m_name: np.concatenate(result[m_name], axis=0) for m_name in methods
    }
    return LinePlotResult(res_data, mask_range)
