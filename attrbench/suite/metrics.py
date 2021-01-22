import torch
import numpy as np
from attrbench import functional


class Metric:
    def __init__(self, model, methods):
        self.model = model
        self.methods = methods
        self.results = {method_name: [] for method_name in methods}
        self.metadata = {}

    def run_batch(self, samples, labels):
        """
        Runs the metric for a given batch, for all methods, and saves result internally
        """
        for method_name in self.methods:
            method = self.methods[method_name]
            self.results[method_name].append(self._run_single_method(samples, labels, method))

    def get_results(self):
        """
        Returns the complete results for all batches and all methods in a dictionary
        """
        result = {}
        shape = None
        for method_name in self.methods:
            result[method_name] = torch.cat(self.results[method_name], dim=0).numpy()
            if shape is None:
                shape = result[method_name].shape
            elif result[method_name].shape != shape:
                raise ValueError(f"Inconsistent shapes for results: "
                                 f"{method_name} had {result[method_name].shape} instead of {shape}")
        return result, shape

    def _run_single_method(self, samples, labels, method):
        raise NotImplementedError


class DeletionUntilFlip(Metric):
    def __init__(self, model, methods, num_steps, masking_policy):
        super().__init__(model, methods)
        self.num_steps = num_steps
        self.masking_policy = masking_policy

    def _run_single_method(self, samples, labels, method):
        return functional.deletion_until_flip(samples, labels, self.model, method, self.num_steps, self.masking_policy).reshape(-1, 1)


class ImpactCoverage(Metric):
    def __init__(self, model, methods, patch_folder):
        super().__init__(model, methods)
        self.patch_folder = patch_folder
        self.attacked_samples, self.patch_mask, self.targets = None, None, None

    def run_batch(self, samples, labels):
        """
        Runs the metric for a given batch, for all methods, and saves result internally
        """
        self.attacked_samples, self.patch_mask, self.targets = functional.apply_patches(samples, labels,
                                                                                        self.model, self.patch_folder)
        for method_name in self.methods:
            method = self.methods[method_name]
            self.results[method_name].append(self._run_single_method(samples, labels, method))

    def _run_single_method(self, samples, labels, method):
        iou = functional.impact_coverage(samples, labels, self.model, method,
                                         attacked_samples=self.attacked_samples,
                                         patch_mask=self.patch_mask,
                                         targets=self.targets)
        return iou.reshape(-1, 1)


class ImpactScore(Metric):
    def __init__(self, model, methods, num_steps, strict, masking_policy, tau=None):
        super().__init__(model, methods)
        self.num_steps = num_steps
        self.strict = strict
        self.masking_policy = masking_policy
        self.tau = tau

    def _run_single_method(self, samples, labels, method):
        return functional.impact_score(samples, labels, self.model, method, self.num_steps,
                                       self.strict, self.masking_policy, self.tau)

    def get_results(self):
        result = {}
        shape = None
        for method_name in self.methods:
            flipped = torch.stack([item[0] for item in self.results[method_name]], dim=0).float()
            totals = torch.tensor([item[1] for item in self.results[method_name]]).reshape(-1, 1).float()
            ratios = flipped / totals
            result[method_name] = ratios.mean(dim=0).numpy().reshape(1, -1)
            if shape is None:
                shape = result[method_name].shape
            elif result[method_name].shape != shape:
                raise ValueError(f"Inconsistent shapes for results: "
                                 f"{method_name} had {result[method_name].shape} instead of {shape}")
        return result, shape


class Insertion(Metric):
    def __init__(self, model, methods, num_steps, masking_policy):
        super().__init__(model, methods)
        self.num_steps = num_steps
        self.masking_policy = masking_policy

    def _run_single_method(self, samples, labels, method):
        return functional.insertion(samples, labels, self.model, method, self.num_steps, self.masking_policy)


class Deletion(Metric):
    def __init__(self, model, methods, num_steps, masking_policy):
        super().__init__(model, methods)
        self.num_steps = num_steps
        self.masking_policy = masking_policy

    def _run_single_method(self, samples, labels, method):
        return functional.deletion(samples, labels, self.model, method, self.num_steps, self.masking_policy)


class Infidelity(Metric):
    def __init__(self, model, methods, perturbation_mode, perturbation_size, num_perturbations):
        super().__init__(model, methods)
        self.perturbation_mode = perturbation_mode
        self.perturbation_size = perturbation_size
        self.num_perturbations = num_perturbations

    def _run_single_method(self, samples, labels, method):
        return functional.infidelity(samples, labels, self.model, method,
                                     self.perturbation_mode, self.perturbation_size, self.num_perturbations)


class MaxSensitivity(Metric):
    def __init__(self, model, methods, radius, num_perturbations):
        super().__init__(model, methods)
        self.radius = radius
        self.num_perturbations = num_perturbations

    def _run_single_method(self, samples, labels, method):
        return functional.max_sensitivity(samples, labels, method, self.radius, self.num_perturbations)


class SensitivityN(Metric):
    def __init__(self, model, methods, min_subset_size, max_subset_size, num_steps, num_subsets, masking_policy):
        super().__init__(model, methods)
        self.min_subset_size = min_subset_size
        self.max_subset_size = max_subset_size
        self.num_steps = num_steps
        self.num_subsets = num_subsets
        self.masking_policy = masking_policy
        self.metadata = {
            "col_index": np.linspace(min_subset_size, max_subset_size, num_steps)
        }

    def _run_single_method(self, samples, labels, method):
        return functional.sensitivity_n(samples, labels, self.model, method,
                                        self.min_subset_size, self.max_subset_size, self.num_steps,
                                        self.num_subsets, self.masking_policy)
