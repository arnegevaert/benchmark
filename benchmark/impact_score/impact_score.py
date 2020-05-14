import numpy as np
from typing import Iterable, List, Dict, Callable


def impact_score(data: Iterable, model: Callable, mask_range: List[int], methods: Dict[str, Callable],
                 mask_value: float, tau: float, device: str = "cpu"):
    original_predictions = []
    method_predictions = {m_name: {n: [] for n in mask_range} for m_name in methods}

    for b, (samples, labels) in enumerate(data):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        original_predictions.append(model(samples).detach().cpu().numpy())  # collect original predictions
        for m_name in methods:
            masked_samples = samples.clone()
            # Get and sort original attributions
            # TODO targets should be the actual predictions of the model, not the ground truth?
            attrs = methods[m_name](samples, target=labels)
            pixel_level_attrs = len(attrs.shape) == 3
            attrs = attrs.reshape(attrs.shape[0], -1)
            sorted_indices = attrs.argsort().cpu()
            for n in mask_range:
                to_mask = sorted_indices[:, -n:]  # [batch_size, i]
                batch_dim = np.column_stack([range(samples.shape[0]) for _ in range(n)])
                if pixel_level_attrs:
                    unraveled = np.unravel_index(to_mask, samples.shape[2:])
                    masked_samples[(batch_dim, -1, *unraveled)] = mask_value
                else:
                    unraveled = np.unravel_index(to_mask, samples.shape[1:])
                    masked_samples[(batch_dim, *unraveled)] = mask_value
                pred = model(masked_samples)
                method_predictions[m_name][n].append(pred.detach().cpu().numpy())

    original_predictions = np.vstack(original_predictions)  # to np array of shape (#images, #classes)
    original_pred_labels = original_predictions.argmax(axis=1)

    method_pred_labels = {}
    # collect labels and raw predictions in single list for each method and masking level
    for m in method_predictions:
        method_predictions[m] = {k: np.vstack([batch_pred for batch_pred in v]) for k, v in method_predictions[m].items()}
        method_pred_labels[m] = {k: pred.argmax(axis=1) for k, pred in method_predictions[m].items()}

    i_score = {}
    i_strict_score = {}
    for m in method_predictions:
        strict_scores = []
        scores = []
        for mask_value, labels in method_pred_labels[m].items():
            # I_{strict} = 1/n * sum_n (y_i' \neq y_i)
            strict_bools = labels != original_pred_labels
            strict_scores.append(np.count_nonzero(strict_bools) / len(strict_bools))

            predictions = method_predictions[m][mask_value]
            # I = 1/n * sum_n ((y_i' \neq y_i) \vee (z_i' \leq z_i)
            bools = original_predictions[np.arange(len(original_pred_labels)), original_pred_labels] * tau >= \
                    predictions[np.arange(len(labels)), labels]
            scores.append(np.count_nonzero(strict_bools + bools) / len(bools))
        i_strict_score[m] = np.array(strict_scores)
        i_score[m] = np.array(scores)
    return i_score, i_strict_score
