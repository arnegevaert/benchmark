import torch
from attrbench import functional


class Metric:
    def __init__(self, model):
        self.model = model
        self.results = {}
        self.metadata = {}

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
    def __init__(self, model, step_size, masking_policy):
        super().__init__(model)
        self.step_size = step_size
        self.masking_policy = masking_policy

    def _run_single_method(self, samples, labels, attrs):
        return functional.deletion_until_flip(samples, labels, self.model, attrs, self.step_size,
                                              self.masking_policy).reshape(-1, 1)


class ImpactCoverage(Metric):
    def __init__(self, model, patch, target_label, ):
        super().__init__(model, methods)
        self.patch = torch.load(patch) if type(patch) == str else patch
        self.target_label = target_label

    def _run_single_method(self, samples, labels, attrs):
        iou, keep = functional.impact_coverage(samples, labels, self.model, attrs, self.patch, self.target_label)
        return iou.reshape(-1, 1)


class ImpactScore(Metric):
    def __init__(self, model, mask_range, strict, masking_policy, tau=None):
        super().__init__(model)
        self.mask_range = mask_range
        self.strict = strict
        self.masking_policy = masking_policy
        self.tau = tau
        self.metadata = {
            "col_index": mask_range
        }

    def _run_single_method(self, samples, labels, attrs):
        return functional.impact_score(samples, labels, self.model, attrs, self.mask_range,
                                       self.strict, self.masking_policy, self.tau)

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
    def __init__(self, model, mask_range, masking_policy):
        super().__init__(model)
        self.mask_range = mask_range
        self.masking_policy = masking_policy
        self.metadata = {
            "col_index": mask_range
        }

    def _run_single_method(self, samples, labels, attrs):
        return functional.insertion(samples, labels, self.model, attrs, self.mask_range, self.masking_policy)


class Deletion(Metric):
    def __init__(self, model, mask_range, masking_policy):
        super().__init__(model)
        self.mask_range = mask_range
        self.masking_policy = masking_policy
        self.metadata = {
            "col_index": mask_range
        }

    def _run_single_method(self, samples, labels, attrs):
        return functional.deletion(samples, labels, self.model, attrs, self.mask_range, self.masking_policy)


class Infidelity(Metric):
    def __init__(self, model, perturbation_range, num_perturbations):
        super().__init__(model)
        self.perturbation_range = perturbation_range
        self.num_perturbations = num_perturbations
        self.metadata = {
            "col_index": perturbation_range
        }

    def _run_single_method(self, samples, labels, attrs):
        return functional.infidelity(samples, labels, self.model, attrs,
                                     self.perturbation_range, self.num_perturbations)


class MaxSensitivity(Metric):
    def __init__(self, model, methods, perturbation_range, num_perturbations):
        super().__init__(model)
        self.methods = methods
        self.results = {method_name: [] for method_name in methods}
        self.perturbation_range = perturbation_range
        self.num_perturbations = num_perturbations
        self.metadata = {
            "col_index": perturbation_range
        }

    def run_batch(self, samples, labels, attrs_dict: dict):
        """
        Runs the metric for a given batch, for all methods, and saves result internally
        """
        for method_name in self.methods:
            method = self.methods[method_name]
            self.results[method_name].append(self._run_single_method(samples, labels, method, attrs_dict[method_name]))

    def _run_single_method(self, samples, labels, method, attrs):
        return functional.max_sensitivity(samples, labels, method, self.perturbation_range, self.num_perturbations,
                                          attrs)


class SensitivityN(Metric):
    def __init__(self, model, methods, n_range, num_subsets, masking_policy):
        super().__init__(model, methods)
        self.n_range = n_range
        self.num_subsets = num_subsets
        self.masking_policy = masking_policy
        self.metadata = {
            "col_index": n_range
        }

    def _run_single_method(self, samples, labels, attrs):
        return functional.sensitivity_n(samples, labels, self.model, attrs,
                                        self.n_range, self.num_subsets, self.masking_policy)
