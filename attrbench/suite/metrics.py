import torch
import numpy as np
from attrbench import functional


class Metric:
    def __init__(self, model, writer=None):
        self.model = model
        self.results = {}
        self.metadata = {}
        self.writer = writer

    def run_batch(self, samples, labels, attrs_dict: dict):
        """
        Runs the metric for a given batch, for all methods, and saves result internally
        """

        for method_name in attrs_dict:
            if method_name not in self.results:
                self.results[method_name] = []
            self.results[method_name].append(self._run_single_method(samples, labels, attrs_dict[method_name]))

    def get_results(self):
        """
        Returns the complete results for all batches and all methods in a dictionary
        """
        result = {}
        shape = None
        for method_name in self.results:
            result[method_name] = torch.cat(self.results[method_name], dim=0).numpy()
            if shape is None:
                shape = result[method_name].shape
            elif result[method_name].shape != shape:
                raise ValueError(f"Inconsistent shapes for results: "
                                 f"{method_name} had {result[method_name].shape} instead of {shape}")
        return result, shape

    def _run_single_method(self, samples, labels, attrs):
        raise NotImplementedError


class DeletionUntilFlip(Metric):
    def __init__(self, model, num_steps, masking_policy, writer=None):
        super().__init__(model, writer)
        self.num_steps = num_steps
        self.masking_policy = masking_policy

    def _run_single_method(self, samples, labels, attrs):
        return functional.deletion_until_flip(samples, self.model, attrs, self.num_steps,
                                              self.masking_policy, writer=self.writer).reshape(-1, 1)


class ImpactCoverage(Metric):
    def __init__(self, model, methods, patch_folder, writer=None):
        super().__init__(model, writer)
        self.methods = methods
        self.results = {method_name: [] for method_name in methods}
        self.patch_folder = patch_folder

    def run_batch(self, samples, labels, attrs_dict=None):
        """
        Runs the metric for a given batch, for all methods, and saves result internally
        """
        attacked_samples, patch_mask, targets = functional.apply_patches(samples, labels,
                                                                         self.model, self.patch_folder)
        for method_name in self.methods:
            method = self.methods[method_name]
            iou = functional.impact_coverage(samples, labels, self.model, method,
                                             attacked_samples=attacked_samples,
                                             patch_mask=patch_mask,
                                             targets=targets,
                                             writer=self.writer).reshape(-1, 1)
            self.results[method_name].append(iou)

    def _run_single_method(self, samples, labels, method, attacked_samples=None, patch_mask=None, targets=None):
        raise NotImplementedError


class ImpactScore(Metric):
    def __init__(self, model, num_steps, strict, masking_policy, tau=None, writer=None):
        super().__init__(model, writer)
        self.num_steps = num_steps
        self.strict = strict
        self.masking_policy = masking_policy
        self.tau = tau

    def _run_single_method(self, samples, labels, attrs):
        return functional.impact_score(samples, labels, self.model, attrs, self.num_steps,
                                       self.strict, self.masking_policy, self.tau,
                                       writer=self.writer)

    def get_results(self):
        result = {}
        shape = None
        for method_name in self.results:
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
    def __init__(self, model, num_steps, masking_policy, writer=None):
        super().__init__(model, writer)
        self.num_steps = num_steps
        self.masking_policy = masking_policy

    def _run_single_method(self, samples, labels, attrs):
        return functional.insertion(samples, labels, self.model, attrs, self.num_steps, self.masking_policy,
                                    writer=self.writer)


class Deletion(Metric):
    def __init__(self, model, num_steps, masking_policy, writer=None):
        super().__init__(model, writer)
        self.num_steps = num_steps
        self.masking_policy = masking_policy

    def _run_single_method(self, samples, labels, attrs):
        return functional.deletion(samples, labels, self.model, attrs, self.num_steps, self.masking_policy,
                                   writer=self.writer)


class Infidelity(Metric):
    def __init__(self, model, perturbation_mode, perturbation_size, num_perturbations, writer=None):
        super().__init__(model, writer)
        self.perturbation_mode = perturbation_mode
        self.perturbation_size = perturbation_size
        self.num_perturbations = num_perturbations

    def _run_single_method(self, samples, labels, attrs):
        return functional.infidelity(samples, labels, self.model, attrs,
                                     self.perturbation_mode, self.perturbation_size, self.num_perturbations,
                                     writer=self.writer)


class MaxSensitivity(Metric):
    def __init__(self, model, methods, radius, num_perturbations, writer=None):
        super().__init__(model, writer)
        self.methods = methods
        self.results = {method_name: [] for method_name in methods}
        self.radius = radius
        self.num_perturbations = num_perturbations

    def run_batch(self, samples, labels, attrs_dict: dict):
        """
        Runs the metric for a given batch, for all methods, and saves result internally
        """
        for method_name in self.methods:
            method = self.methods[method_name]
            max_sens = functional.max_sensitivity(samples, labels, method, attrs_dict[method_name], self.radius,
                                                  self.num_perturbations, writer=self.writer)
            self.results[method_name].append(max_sens)

    def _run_single_method(self, samples, labels, attrs):
        raise NotImplementedError


class SensitivityN(Metric):
    def __init__(self, model, min_subset_size, max_subset_size, num_steps, num_subsets, masking_policy, writer=None):
        super().__init__(model, writer)
        self.min_subset_size = min_subset_size
        self.max_subset_size = max_subset_size
        self.num_steps = num_steps
        self.num_subsets = num_subsets
        self.masking_policy = masking_policy
        self.metadata = {
            "col_index": np.linspace(min_subset_size, max_subset_size, num_steps)
        }

    def _run_single_method(self, samples, labels, attrs):
        return functional.sensitivity_n(samples, labels, self.model, attrs,
                                        self.min_subset_size, self.max_subset_size, self.num_steps,
                                        self.num_subsets, self.masking_policy,
                                        writer=self.writer)
