from typing import Iterable, Callable, List, Dict
import numpy as np
from tqdm import tqdm
from attrbench.evaluation.result import LinePlotResult
from attrbench.evaluation import util
import torch
import torch.nn.functional as F


_OUTPUT_TRANSFORMS = {
    "identity": lambda l: l,
    "softmax": lambda l: F.softmax(l, dim=1),
    "logit_softmax": lambda l: util.logit_softmax(l)
}


def insertion_deletion_curves(data: Iterable, model: Callable, methods: Dict[str, Callable],
                              mask_range: List[int], mask_value: float, pixel_level_mask: bool,
                              device: str, mode: str, output_transform: str = "identity"):
    assert output_transform in ["identity", "softmax", "logit_softmax"]
    assert mode in ["deletion", "insertion"]
    result = {m_name: [] for m_name in methods}
    full_mask_output = None
    for batch_index, (samples, labels) in enumerate(tqdm(data)):
        if full_mask_output is None:
            full_mask_output = model(torch.ones(samples.shape).to(device) * mask_value)
            full_mask_output = _OUTPUT_TRANSFORMS[output_transform](full_mask_output)
        samples = samples.to(device)
        labels = labels.to(device)
        # Check which samples are classified correctly
        # Only want to calculate for correctly classified samples
        with torch.no_grad():
            y_pred = torch.argmax(model(samples), dim=1)
        samples = samples[y_pred == labels]
        labels = labels[y_pred == labels]
        batch_size = samples.size(0)
        color_channels = samples.size(1)
        if batch_size > 0:
            for key in methods:
                batch_result = []
                attrs = methods[key](samples, labels)  # [batch_size, *sample_shape]
                # Flatten each sample in order to sort indices per sample
                attrs = attrs.flatten(1)  # [batch_size, -1]
                # Sort indices of attrs in ascending order
                sorted_indices = attrs.argsort().cpu().detach().numpy()
                orig_predictions = _OUTPUT_TRANSFORMS[output_transform](model(samples))
                orig_predictions = orig_predictions[np.arange(batch_size), labels].reshape(-1, 1)

                # Add first value to result (full mask or no mask)
                if mode == "deletion":
                    batch_result.append(orig_predictions.cpu().detach().numpy())
                else:
                    full_mask_predictions = full_mask_output[np.arange(batch_size), labels].reshape(-1, 1)
                    batch_result.append(full_mask_predictions.cpu().detach().numpy())

                # Add partially masked values to result
                for i in mask_range:
                    # Get indices of i most important inputs
                    masked_samples = samples.clone() if mode == "deletion" \
                        else torch.ones(samples.shape).to(device) * mask_value
                    to_mask = sorted_indices[:, -i:]  # [batch-size, i]
                    batch_dim = np.column_stack([range(batch_size) for _ in range(i)])
                    color_dim = np.column_stack([range(color_channels) for _ in range(i)])
                    if pixel_level_mask:
                        unraveled = np.unravel_index(to_mask, samples.shape[2:])
                        masked_samples[(batch_dim, color_dim, *unraveled)] = mask_value if mode == "deletion" \
                            else samples[(batch_dim, color_dim, *unraveled)]
                    else:
                        unraveled = np.unravel_index(to_mask, samples.shape[1:])
                        masked_samples[(batch_dim, *unraveled)] = mask_value if mode == "deletion" \
                            else samples[(batch_dim, *unraveled)]
                    # Get predictions for result
                    predictions = _OUTPUT_TRANSFORMS[output_transform](model(masked_samples))
                    predictions = predictions[np.arange(batch_size), labels].reshape(-1, 1)
                    batch_result.append(predictions.cpu().detach().numpy())

                # Add final value to result (full mask or no mask)
                if mode == "insertion":
                    batch_result.append(orig_predictions.cpu().detach().numpy())
                else:
                    full_mask_predictions = full_mask_output[np.arange(batch_size), labels].reshape(-1, 1)
                    batch_result.append(full_mask_predictions.cpu().detach().numpy())
                batch_result = np.concatenate(batch_result, axis=1)
                result[key].append(batch_result)
    res_data = {
        # [n_batches*batch_size, len(mask_range)]
        m_name: np.concatenate(result[m_name], axis=0) for m_name in methods
    }
    return Result(res_data, mask_range, mode)


# TODO THIS NORMALIZATION DOES MAKE A DIFFERENCE IN THE ACTUAL PLOT
# TODO see which mode is most reasonable
class Result(LinePlotResult):
    def __init__(self, data, x_range, mode):
        super().__init__(data, x_range)
        normalized = {}
        for method in self.processed:
            normalized[method] = {}
            for key in ["mean", "lower", "upper"]:
                arr = self.processed[method][key]
                if mode == "deletion":
                    normalized[method][key] = (arr - arr[-1]) / (arr[0] - arr[-1])
                else:
                    normalized[method][key] = (arr - arr[0]) / (arr[-1] - arr[0])
        self.processed = normalized
