from typing import Callable, List, Union, Tuple, Dict
from os import path

import numpy as np
import torch

from attrbench.lib import AttributionWriter
from attrbench.metrics import Metric
from ._compute_perturbations import _compute_perturbations
from ._compute_result import _compute_result
from . import perturbation_generator
from .result import InfidelityResult
import time
import logging


def infidelity(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
               pert_generator: perturbation_generator.PerturbationGenerator, num_perturbations: int,
               loss_fns: Union[Tuple[str], str] = "mse", activation_fns: Union[Tuple[str], str] = "linear",
               writer: AttributionWriter = None) -> Dict:
    if type(activation_fns) == str:
        activation_fns = (activation_fns,)
    pert_vectors, pred_diffs = _compute_perturbations(samples, labels, model, pert_generator,
                                                      num_perturbations, activation_fns, writer)
    if type(loss_fns) == str:
        loss_fns = (loss_fns,)
    for m in loss_fns:
        if m not in ("mse", "corr"):
            raise ValueError(f"Invalid mode: {m}")
    return _compute_result(pert_vectors, pred_diffs, attrs, loss_fns)


def _parse_pert_generator(d):
    constructor = getattr(perturbation_generator, d["type"])
    return constructor(**{key: value for key, value in d.items() if key != "type"})


class Infidelity(Metric):
    def __init__(self, model: Callable, method_names: List[str], perturbation_generators: Dict,
                 num_perturbations: int,
                 loss_fns: Union[Tuple[str], str] = "mse",
                 activation_fns: Union[Tuple[str], str] = "linear", writer_dir: str = None):
        super().__init__(model, method_names)  # We don't pass writer_dir to super because we only use 1 general writer
        self.writers = {"general": AttributionWriter(path.join(writer_dir, "general"))} \
            if writer_dir is not None else None
        self.num_perturbations = num_perturbations
        self.loss_fns = (loss_fns,) if type(loss_fns) == str else loss_fns
        self.activation_fns = (activation_fns,) if type(activation_fns) == str else activation_fns
        # Process "perturbation-generators" argument: either it is a dictionary of PerturbationGenerator objects,
        # or it is a dictionary that needs to be parsed.
        self.perturbation_generators = {}
        for key, value in perturbation_generators.items():
            if type(value) == perturbation_generator.PerturbationGenerator:
                self.perturbation_generators[key] = value
            else:
                self.perturbation_generators[key] = _parse_pert_generator(value)

        self.result: InfidelityResult = InfidelityResult(method_names, list(perturbation_generators.keys()),
                                                         list(self.activation_fns),
                                                         list(self.loss_fns))

    def run_batch(self, samples, labels, attrs_dict: dict, baseline_attrs: np.ndarray):
        # First calculate perturbation vectors and predictions differences, these can be re-used for all methods
        writer = self.writers["general"] if self.writers is not None else None

        pert_vectors, pred_diffs = {}, {}
        for key, pert_gen in self.perturbation_generators.items():
            p_vectors, p_diffs = _compute_perturbations(samples, labels, self.model, pert_gen,
                                                        self.num_perturbations, self.activation_fns,
                                                        writer)
            pert_vectors[key] = p_vectors
            pred_diffs[key] = p_diffs

        # Compute and append results
        baseline_results = {
            pert_gen: {
                loss: {
                    afn: [] for afn in pred_diffs[pert_gen].keys()
                } for loss in self.loss_fns} for pert_gen in pred_diffs.keys()}
        method_results = {}
        for pert_gen in self.perturbation_generators:
            # Calculate baseline results
            for i in range(baseline_attrs.shape[0]):
                baseline_result = _compute_result(pert_vectors[pert_gen], pred_diffs[pert_gen], baseline_attrs[i, ...],
                                    self.loss_fns)
                for loss in baseline_result.keys():
                    for afn in baseline_result[loss].keys():
                        baseline_results[pert_gen][loss][afn].append(baseline_result[loss][afn])
            for loss in baseline_results[pert_gen].keys():
                for afn in baseline_results[pert_gen][loss].keys():
                    baseline_results[pert_gen][loss][afn] = np.stack(baseline_results[pert_gen][loss][afn], axis=1)

            # Calculate actual method results
            method_results[pert_gen] = {}
            for method_name in attrs_dict.keys():
                method_results[pert_gen][method_name] = _compute_result(pert_vectors[pert_gen], pred_diffs[pert_gen],
                                                                        attrs_dict[method_name], self.loss_fns)
        self.result.append(method_results, baseline_results)
        logging.info(f"Appended Infidelity")

    def get_result(self) -> InfidelityResult:
        return self.result
