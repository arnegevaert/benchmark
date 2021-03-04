import torch
import numpy as np
from attrbench import functional
from attrbench.lib import AttributionWriter
from os import path


class Metric:
    def __init__(self, model, writer_dir=None):
        self.model = model
        self.results = {}
        self.metadata = {}
        self.writer_dir = writer_dir
        self.writers = {}

    def run_batch(self, samples, labels, attrs_dict: dict):
        """
        Runs the metric for a given batch, for all methods, and saves result internally
        """
        for method_name in attrs_dict:
            if method_name not in self.results:
                self.results[method_name] = []
            self.results[method_name].append(self._run_single_method(samples, labels, attrs_dict[method_name],
                                                                     writer=self._get_writer(method_name)))

    def _get_writer(self, method_name):
        if method_name not in self.writers:
            if self.writer_dir is None:
                self.writers[method_name] = None
            else:
                self.writers[method_name] = AttributionWriter(path.join(self.writer_dir, method_name))
        writer = self.writers[method_name]
        if writer:
            writer.set_method_name(method_name)
            writer.increment_batch()
        return writer

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

    def _run_single_method(self, samples, labels, attrs: np.ndarray, writer=None):
        raise NotImplementedError


class DeletionUntilFlip(Metric):
    def __init__(self, model, num_steps, masker, writer_dir=None):
        super().__init__(model, writer_dir)
        self.num_steps = num_steps
        self.masker = masker

    def _run_single_method(self, samples, labels, attrs, writer=None):
        return functional.deletion_until_flip(samples, self.model, attrs, self.num_steps,
                                              self.masker, writer=writer).reshape(-1, 1)


class ImpactCoverage(Metric):
    def __init__(self, model, methods, patch_folder, writer_dir=None):
        super().__init__(model, writer_dir)
        self.methods = methods
        self.results = {method_name: [] for method_name in methods}
        self.patch_folder = patch_folder
        self.writers = {method_name: path.join(writer_dir, method_name) if writer_dir else None for method_name in
                        methods}

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
                                             writer=self._get_writer(method_name)).reshape(-1, 1)
            self.results[method_name].append(iou)

    def _run_single_method(self, samples, labels, method, attacked_samples=None, patch_mask=None, targets=None):
        raise NotImplementedError


class ImpactScore(Metric):
    def __init__(self, model, num_steps, strict, masker, tau=None, writer_dir=None):
        super().__init__(model, writer_dir)
        self.num_steps = num_steps
        self.strict = strict
        self.masker = masker
        self.tau = tau

    def _run_single_method(self, samples, labels, attrs, writer=None):
        return functional.impact_score(samples, labels, self.model, attrs, self.num_steps,
                                       self.strict, self.masker, self.tau,
                                       writer=writer)

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
    def __init__(self, model, num_steps, masker, writer_dir=None):
        super().__init__(model, writer_dir)
        self.num_steps = num_steps
        self.masker = masker

    def _run_single_method(self, samples, labels, attrs: np.ndarray, writer=None):
        return functional.insertion(samples, labels, self.model, attrs, self.num_steps, self.masker,
                                    writer=writer)


class Deletion(Metric):
    def __init__(self, model, num_steps, masker, writer_dir=None):
        super().__init__(model, writer_dir)
        self.num_steps = num_steps
        self.masker = masker

    def _run_single_method(self, samples, labels, attrs: np.ndarray, writer=None):
        return functional.deletion(samples, labels, self.model, attrs, self.num_steps, self.masker,
                                   writer=writer)


class Infidelity(Metric):
    def __init__(self, model, perturbation_mode, perturbation_size, num_perturbations, writer_dir=None):
        super().__init__(model, writer_dir)
        self.perturbation_mode = perturbation_mode
        self.perturbation_size = perturbation_size
        self.num_perturbations = num_perturbations

    def run_batch(self, samples, labels, attrs_dict: dict):
        # First calculate perturbation vectors and predictions differences, these can be re-used for all methods
        pert_vectors, pred_diffs = functional.infid_perturbations(samples, labels, self.model,
                                                                  self.perturbation_mode, self.perturbation_size,
                                                                  self.num_perturbations)
        for method_name in attrs_dict:
            if method_name not in self.results:
                self.results[method_name] = []
            self.results[method_name].append(functional.infid_mse(pert_vectors, pred_diffs, attrs_dict[method_name]))


class MaxSensitivity(Metric):
    def __init__(self, model, methods, radius, num_perturbations, writer_dir=None):
        super().__init__(model, writer_dir)
        self.methods = methods
        self.results = {method_name: [] for method_name in methods}
        self.radius = radius
        self.num_perturbations = num_perturbations
        self.writers = {method_name: path.join(writer_dir, method_name) if writer_dir else None for method_name in
                        methods}

    def run_batch(self, samples, labels, attrs_dict: dict):
        """
        Runs the metric for a given batch, for all methods, and saves result internally
        """
        for method_name in self.methods:
            method = self.methods[method_name]
            max_sens = functional.max_sensitivity(samples, labels, method, attrs_dict[method_name], self.radius,
                                                  self.num_perturbations, writer=self._get_writer(method_name))
            self.results[method_name].append(max_sens)

    def _run_single_method(self, samples, labels, attrs, writer=None):
        raise NotImplementedError


class SensitivityN(Metric):
    def __init__(self, model, min_subset_size, max_subset_size, num_steps, num_subsets, masker, writer_dir=None):
        super().__init__(model, writer_dir)
        self.min_subset_size = min_subset_size
        self.max_subset_size = max_subset_size
        self.num_steps = num_steps
        self.num_subsets = num_subsets
        self.masker = masker
        self.metadata = {
            "col_index": np.linspace(min_subset_size, max_subset_size, num_steps)
        }

    def run_batch(self, samples, labels, attrs_dict: dict):
        # Get total number of features from attributions dict
        attrs = attrs_dict[next(iter(attrs_dict))]
        num_features = attrs.reshape(attrs.shape[0], -1).shape[1]
        # Calculate n_range
        # TODO it should be possible to do this in the constructor
        n_range = (np.linspace(self.min_subset_size, self.max_subset_size, self.num_steps) * num_features).astype(np.int)
        # Create pseudo-dataset
        ds = functional.SensitivityNDataset(n_range, self.num_subsets, samples.cpu().numpy(), num_features, self.masker)
        # Calculate output diffs and removed indices (we will re-use this for each method)
        output_diffs, indices = functional.sens_n_perturbations(samples, labels, ds, self.model, n_range)

        for method_name in attrs_dict:
            if method_name not in self.results:
                self.results[method_name] = []

            attrs = attrs_dict[method_name]
            attrs = attrs.reshape(attrs.shape[0], 1, -1)  # [batch_size, 1, -1]
            method_result = []
            for n in n_range:
                # Calculate sums of attributions
                mask_attrs = np.take_along_axis(attrs, axis=-1, indices=indices[n])  # [batch_size, num_subsets, n]
                sum_of_attrs = mask_attrs.sum(axis=-1)  # [batch_size, num_subsets]
                method_result.append(functional.sens_n_correlations(sum_of_attrs, output_diffs[n]))
            method_result = torch.tensor(np.stack(method_result, axis=1))
            self.results[method_name].append(method_result)


class SegSensN(Metric):
    def __init__(self, model, masker, min_subset_size, max_subset_size, num_steps, num_subsets, writer_dir=None):
        super().__init__(model, writer_dir)
        self.min_subset_size = min_subset_size
        self.max_subset_size = max_subset_size
        self.num_steps = num_steps
        self.num_subsets = num_subsets
        # Total number of segments is fixed 100
        self.n_range = (np.linspace(self.min_subset_size, self.max_subset_size, self.num_steps) * 100).astype(np.int)
        self.masker = masker
        self.metadata = {
            "col_index": np.linspace(min_subset_size, max_subset_size, num_steps)
        }

    def run_batch(self, samples, labels, attrs_dict: dict):
        # Create pseudo-dataset
        ds = functional.SegSensNDataset(self.n_range, self.num_subsets, samples.cpu().numpy(), self.masker)
        # Calculate output diffs and removed indices (we will re-use this for each method)
        output_diffs, indices = functional.sens_n_perturbations(samples, labels, ds, self.model, self.n_range)

        for method_name in attrs_dict:
            if method_name not in self.results:
                self.results[method_name] = []

            attrs = attrs_dict[method_name]
            attrs = attrs.reshape(attrs.shape[0], 1, -1)  # [batch_size, 1, -1]
            method_result = []
            for n in self.n_range:
                # Calculate sums of attributions
                mask_attrs = np.take_along_axis(attrs, axis=-1, indices=indices[n])  # [batch_size, num_subsets, n]
                sum_of_attrs = mask_attrs.sum(axis=-1)  # [batch_size, num_subsets]
                method_result.append(functional.sens_n_correlations(sum_of_attrs, output_diffs[n]))
            method_result = torch.tensor(np.stack(method_result, axis=1))
            self.results[method_name].append(method_result)


class IROF(Metric):
    def __init__(self, model, masker, writer_dir=None):
        super().__init__(model, writer_dir)
        self.masker = masker

    def _run_single_method(self, samples, labels, attrs, writer=None):
        return functional.irof(samples, labels, self.model, attrs, self.masker, writer)


class IIOF(Metric):
    def __init__(self, model, masker, writer_dir=None):
        super().__init__(model, writer_dir)
        self.masker = masker

    def _run_single_method(self, samples, labels, attrs, writer=None):
        return functional.iiof(samples, labels, self.model, attrs, self.masker, writer)
