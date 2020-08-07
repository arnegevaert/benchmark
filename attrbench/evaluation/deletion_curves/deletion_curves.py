from typing import Iterable, Callable, List, Dict
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


def deletion_curves(data: Iterable, model: Callable, methods: Dict[str, Callable],
                    mask_range: List[int], mask_value: float, pixel_level_mask: bool,
                    device: str):
    result = {m_name: [] for m_name in methods}
    for batch_index, (samples, labels) in enumerate(tqdm(data)):
        samples = samples.to(device)
        labels = labels.to(device)
        for key in methods:
            batch_result = []
            attrs = methods[key](samples, labels)  # [batch_size, *sample_shape]
            # Flatten each sample in order to sort indices per sample
            attrs = attrs.flatten(1)  # [batch_size, -1]
            # Sort indices of attrs in ascending order
            sorted_indices = attrs.argsort().cpu().detach().numpy()
            # TODO create all masked samples at once for more efficient GPU usage
            for i in mask_range:
                # Get indices of i most important inputs
                to_mask = sorted_indices[:, -i:]  # [batch-size, i]
                masked_samples = samples.clone()
                batch_dim = np.column_stack([range(samples.shape[0]) for _ in range(i)])
                if pixel_level_mask:
                    unraveled = np.unravel_index(to_mask, samples.shape[2:])
                    masked_samples[(batch_dim, -1, *unraveled)] = mask_value
                else:
                    unraveled = np.unravel_index(to_mask, samples.shape[1:])
                    masked_samples[(batch_dim, *unraveled)] = mask_value
                # Get predictions for result
                predictions = model(masked_samples)
                predictions = predictions[np.arange(predictions.shape[0]), labels].reshape(-1, 1)
                batch_result.append(predictions.cpu().detach().numpy())
            batch_result = np.concatenate(batch_result, axis=1)
            result[key].append(batch_result)
    # [n_batches*batch_size, len(mask_range)]
    raw_data = {
        "x_range": mask_range,
        "data": {
            m_name: np.concatenate(result[m_name], axis=0) for m_name in methods
        }
    }
    return DeletionCurvesResult(raw_data=raw_data), raw_data


class DeletionCurvesResult:
    def __init__(self, filename=None, raw_data=None):
        if not (raw_data or filename):
            raise ValueError("Must provide raw data dict or file name to load.")
        if raw_data:
            self.processed = {}
            self.x_range = raw_data["x_range"]
            for method in raw_data["data"]:
                normalized = raw_data["data"][method] / np.mean(raw_data["data"][method][:, 0])
                sd = np.std(normalized, axis=0)
                mean = np.mean(normalized, axis=0)
                self.processed[method] = {
                    "mean": mean,
                    "lower": mean - (1.96 * sd / np.sqrt(normalized.shape[0])),
                    "upper": mean + (1.96 * sd / np.sqrt(normalized.shape[0]))
                }
        elif filename:
            with open(filename) as file:
                contents = json.load(file)
                data = contents["data"]
                self.x_range = contents["x_range"]
                self.processed = {
                    method: {
                        stat: np.array(data[method][stat]) for stat in data[method]
                    } for method in data
                }

    def plot(self, interval=False):
        fig, ax = plt.subplots(figsize=(7, 5))
        for method in self.processed:
            ax.plot(self.x_range, self.processed[method]["mean"], label=method)
            if interval:
                ax.fill_between(x=self.x_range, y1=self.processed[method]["lower"],
                                y2=self.processed[method]["upper"], alpha=.2)
        ax.legend(loc=(0., 1.05), ncol=3)
        return fig, ax

    def auc(self):
        return {
            method: {
                "mean": np.mean(self.processed[method]["mean"]),
                "lower": np.mean(self.processed[method]["lower"]),
                "upper": np.mean(self.processed[method]["upper"])
            } for method in self.processed
        }

    def save(self, filename):
        with open(filename, "w") as outfile:
            json.dump({
                "x_range": self.x_range,
                "data": {
                    method: {
                        stat: self.processed[method][stat].tolist() for stat in self.processed[method]
                    } for method in self.processed
                }
            }, outfile)
