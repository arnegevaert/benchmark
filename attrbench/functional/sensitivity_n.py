from typing import Callable
import numpy as np
from attrbench.lib import sum_of_attributions, mask_segments, segment_samples_attributions
from attrbench.lib.masking import Masker
import torch
import warnings


def _calculate_correlations(sum_of_attrs, output_diffs):
    # Calculate correlation between output difference and sum of attribution values
    # Subtract mean
    sum_of_attrs -= sum_of_attrs.mean(dim=1, keepdim=True)
    output_diffs -= output_diffs.mean(dim=1, keepdim=True)
    # Calculate covariances
    cov = (sum_of_attrs * output_diffs).sum(dim=1) / (sum_of_attrs.shape[1] - 1)
    # Divide by product of standard deviations
    # [batch_size]
    denom = sum_of_attrs.std(dim=1) * output_diffs.std(dim=1)
    denom_zero = (denom == 0.)
    if torch.any(denom_zero):
        warnings.warn("Zero standard deviation detected.")
    corrcoefs = cov / (sum_of_attrs.std(dim=1) * output_diffs.std(dim=1))
    corrcoefs[denom_zero] = 0.
    return corrcoefs


def sensitivity_n(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: torch.Tensor,
                  min_subset_size: float, max_subset_size: float, num_steps: int, num_subsets: int,
                  masker: Masker, writer=None):
    device = samples.device
    attrs = attrs.to(device)

    result = []
    batch_size = samples.size(0)
    masker.initialize_baselines(samples)

    with torch.no_grad():
        orig_output = model(samples)

    num_features = attrs.flatten(1).shape[1]
    n_range = (np.linspace(min_subset_size, max_subset_size, num_steps) * num_features).astype(np.int)
    for n in n_range:
        output_diffs = []
        sum_of_attrs = []

        for ns in range(num_subsets):
            # Generate mask and masked samples
            # Mask is generated using replace=False, same mask is used for all samples in batch
            indices = torch.tensor(np.random.choice(num_features, size=n, replace=False)).long().repeat(batch_size, 1)
            output, masked_samples = masker.predict_masked(samples, indices, model, return_masked_samples=True)
            if writer is not None:
                writer.add_images("Masked samples N={}".format(n), masked_samples, global_step=ns)
            # Get difference in output confidence for desired class
            output_diffs.append((orig_output - output).gather(dim=1, index=labels.unsqueeze(-1)))
            # Get sum of attribution values for masked inputs
            sum_of_attrs.append(sum_of_attributions(attrs, indices.to(attrs.device)))
        # [batch_size, num_subsets]
        sum_of_attrs = torch.cat(sum_of_attrs, dim=1)
        output_diffs = torch.cat(output_diffs, dim=1)
        result.append(_calculate_correlations(sum_of_attrs, output_diffs))
    # [batch_size, len(n_range)]
    result = torch.stack(result, dim=1).cpu().detach()
    return result


def seg_sensitivity_n(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: torch.Tensor,
                      min_subset_size: float, max_subset_size: float, num_steps: int, num_subsets: int,
                      masker: Masker, writer=None):
    # Segment images and attributions
    segmented_images, avg_attrs = segment_samples_attributions(samples.detach().cpu().numpy(),
                                                               attrs.detach().cpu().numpy())
    if writer is not None:
        writer.add_images("segmented samples", segmented_images)

    # Initialize masker
    masker.initialize_baselines(samples)

    # Get original predictions
    with torch.no_grad():
        orig_predictions = model(samples).gather(dim=1, index=labels.unsqueeze(-1))

    # Total number of segments is fixed 100
    n_range = (np.linspace(min_subset_size, max_subset_size, num_steps) * 100).astype(np.int)

    valid_indices = [np.where(~np.isinf(avg_attrs[i, ...]))[0] for i in range(samples.size(0))]
    result = []
    for n in n_range:
        output_diffs = []
        sum_of_attrs = []

        for ns in range(num_subsets):
            # Select indices of segments to mask for each sample
            # This needs to happen separately per sample because number of segments varies
            indices = np.stack([np.random.choice(valid_indices[i], size=n, replace=False)
                                for i in range(samples.size(0))])
            # Mask samples and get predictions
            masked_samples = mask_segments(samples, segmented_images, indices, masker)
            if writer is not None:
                writer.add_images("Masked samples N={}".format(n), masked_samples, global_step=ns)
            with torch.no_grad():
                predictions = model(masked_samples).gather(dim=1, index=labels.unsqueeze(-1))
            # Save prediction differences and sum of attrs
            output_diffs.append((orig_predictions - predictions).cpu())
            sum_of_attrs.append(sum_of_attributions(torch.tensor(avg_attrs), torch.tensor(indices)))
        # [batch_size, num_subsets]
        sum_of_attrs = torch.cat(sum_of_attrs, dim=1)
        output_diffs = torch.cat(output_diffs, dim=1)
        result.append(_calculate_correlations(sum_of_attrs, output_diffs))
    # [batch_size, len(n_range)]
    result = torch.stack(result, dim=1).cpu().detach()
    return result
