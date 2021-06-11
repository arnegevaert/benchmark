from typing import Callable, List, Union, Tuple, Dict
from os import path

import numpy as np
import torch

from attrbench.lib import AttributionWriter
from attrbench.metrics import Metric
from ._compute_perturbations import _compute_perturbations
from . import perturbation_generator
from .result import InfidelityResult
import logging

# TODO this is broken
def infidelity(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
               pert_generator: perturbation_generator.PerturbationGenerator, num_perturbations: int,
               activation_fns: Union[Tuple[str], str] = "linear",
               writer: AttributionWriter = None) -> Dict:
    if type(activation_fns) == str:
        activation_fns = (activation_fns,)
    pert_vectors, pred_diffs = _compute_perturbations(samples, labels, model, pert_generator,
                                                      num_perturbations, activation_fns, writer)
    #return _compute_result(pert_vectors, pred_diffs, attrs)


def _parse_pert_generator(d):
    constructor = getattr(perturbation_generator, d["type"])
    return constructor(**{key: value for key, value in d.items() if key != "type"})


class Infidelity(Metric):
    def __init__(self, model: Callable, method_names: List[str], perturbation_generators: Dict,
                 num_perturbations: int,
                 activation_fns: Union[Tuple[str], str] = "linear", writer_dir: str = None):
        super().__init__(model, method_names)  # We don't pass writer_dir to super because we only use 1 general writer
        self.writers = {"general": AttributionWriter(path.join(writer_dir, "general"))} \
            if writer_dir is not None else None
        self.num_perturbations = num_perturbations
        self.activation_fns = (activation_fns,) if type(activation_fns) == str else activation_fns
        # Process "perturbation-generators" argument: either it is a dictionary of PerturbationGenerator objects,
        # or it is a dictionary that needs to be parsed.
        self.perturbation_generators = {}
        for key, value in perturbation_generators.items():
            if type(value) == perturbation_generator.PerturbationGenerator:
                self.perturbation_generators[key] = value
            else:
                self.perturbation_generators[key] = _parse_pert_generator(value)

        self._result: InfidelityResult = InfidelityResult(method_names + ["_BASELINE"],
                                                          list(perturbation_generators.keys()),
                                                          list(self.activation_fns))

    def run_batch(self, samples, labels, attrs_dict: dict, baseline_attrs: np.ndarray):
        # First calculate perturbation vectors and predictions differences, these can be re-used for all methods
        writer = self.writers["general"] if self.writers is not None else None

        for pert_gen, pert_gen_fn in self.perturbation_generators.items():
            # Calculate dot products and prediction differences
            extended_attrs_dict = {key: value for key, value in attrs_dict.items()}
            for i in range(baseline_attrs.shape[0]):
                extended_attrs_dict[f"_BASELINE_{i}"] = baseline_attrs[i, ...]
            result = _compute_perturbations(samples, labels, self.model, extended_attrs_dict,
                                            pert_gen_fn, self.num_perturbations,
                                            self.activation_fns, writer)
            # Append method results
            for method_name in attrs_dict.keys():
                method_result = {afn: result[afn][method_name] for afn in self.activation_fns}
                self.result.append(method_result, perturbation_generator=pert_gen, method=method_name)

            # Append baseline results
            bl_result = {afn: np.stack([result[afn][f"_BASELINE_{i}"] for i in range(baseline_attrs.shape[0])], axis=1)
                         for afn in self.activation_fns}
            self.result.append(bl_result, perturbation_generator=pert_gen, method="_BASELINE")
