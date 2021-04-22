from typing import Callable, List, Union, Tuple, Dict
from os import path
import os
import multiprocessing

import numpy as np
import torch

from attrbench.lib import AttributionWriter
from attrbench.metrics import Metric, MetricResult
from ._compute_perturbations import _compute_perturbations
from ._compute_result import _compute_result
from . import perturbation_generator
from .result import InfidelityResult
import time
import logging
from functools import partial


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
    res = _compute_result(pert_vectors, pred_diffs, {"m": attrs}, loss_fns)
    return res["m"]


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
        self.pool = None

    def run_batch(self, samples, labels, attrs_dict: dict):
        if self.pool is not None:
            start_t = time.time()
            logging.info("Joining...")
            self.pool.join()
            end_t = time.time()
            logging.info(f"Join done in {end_t - start_t:.2f}s")
        # First calculate perturbation vectors and predictions differences, these can be re-used for all methods
        writer = self.writers["general"] if self.writers is not None else None

        for pert_gen in self.perturbation_generators:
            pert_vectors, pred_diffs = _compute_perturbations(samples, labels, self.model, pert_gen,
                                                              self.num_perturbations, self.activation_fns, writer)
            if os.getenv("NO_MULTIPROC"):
                results = _compute_result(pert_vectors, pred_diffs, attrs_dict, self.loss_fns)
                self.result.append(pert_gen, results)
            else:
                self.pool = multiprocessing.pool.ThreadPool(processes=1)
                self.pool.apply_async(_compute_result, args=(pert_vectors, pred_diffs, attrs_dict, self.loss_fns),
                                      callback=lambda res: self.result.append(pert_gen, res))
                self.pool.close()

    def get_result(self) -> InfidelityResult:
        if self.pool is not None:
            start_t = time.time()
            logging.info("Joining...")
            self.pool.join()
            end_t = time.time()
            logging.info(f"Join done in {end_t - start_t:.2f}s")
        return self.result
