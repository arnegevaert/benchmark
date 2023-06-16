from .._metric import Metric
import warnings
from typing import Dict, Tuple
from attribench.data import AttributionsDataset
from ._infidelity_worker import InfidelityWorker
from attribench.result import InfidelityResult
from attribench.functional.metrics.infidelity._perturbation_generator import (
    PerturbationGenerator,
)
from torch import multiprocessing as mp
from attribench._model_factory import ModelFactory
from multiprocessing.synchronize import Event


class Infidelity(Metric):
    def __init__(
        self,
        model_factory: ModelFactory,
        dataset: AttributionsDataset,
        batch_size: int,
        perturbation_generators: Dict[str, PerturbationGenerator],
        num_perturbations: int,
        activation_fns: Tuple[str],
    ):
        """Computes the Infidelity metric for a given `AttributionsDataset` and
        model using multiple processes.

        Infidelity is computed by generating perturbations for each sample in the
        dataset and computing the difference in the model's output on the original
        sample and the perturbed sample. This difference is then compared to the
        dot product of the perturbation vector and the attribution map for each
        attribution method. The Infidelity metric is the mean squared error between
        these two values.

        The idea is that if the dot product is large, then the perturbation vector
        is aligned with the attribution map, and the model's output should change
        significantly when the perturbation is applied. If the dot product is small,
        then the perturbation vector is not aligned with the attribution map, and
        the model's output should not change significantly when the perturbation is
        applied.

        The mean squared error is computed for `num_perturbations` perturbations
        for each sample. The `perturbation_generators` argument is a dictionary
        mapping perturbation generator names to `PerturbationGenerator` objects.
        These objects can be used to implement different versions of Infidelity.

        The Infidelity metric is computed for each perturbation generator in
        `perturbation_generators` and each activation function in `activation_fns`.
        The number of processes is determined by the number of devices. If `devices`
        is None, then all available devices are used. Samples are distributed evenly
        across the processes.

        Parameters
        ----------
        model_factory : ModelFactory
            ModelFactory instance or callable that returns a model.
            Used to create a model for each subprocess.
        dataset : AttributionsDataset
            Dataset containing the samples and attributions to compute
            Infidelity on.
        batch_size : int
            Batch size to use when computing Infidelity.
        perturbation_generators : Dict[str, PerturbationGenerator]
            Dictionary of perturbation generators to use for generating
            perturbations.
        num_perturbations : int
            Number of perturbations to generate for each sample.
        activation_fns : Tuple[str]
            Tuple of activation functions to use when computing Infidelity.
        """
        super().__init__(model_factory, dataset, batch_size)
        if not dataset.group_attributions:
            warnings.warn(
                "Infidelity expects a dataset with group_attributions==True."
                "Setting to True."
            )
            dataset.group_attributions = True
        self.dataset = dataset
        self.activation_fns = activation_fns
        self.num_perturbations = num_perturbations
        self.perturbation_generators = perturbation_generators
        self._result = InfidelityResult(
            self.dataset.method_names,
            tuple(self.perturbation_generators.keys()),
            self.activation_fns,
            shape=(self.dataset.num_samples, 1),
        )

    def _create_worker(
        self, queue: mp.Queue, rank: int, all_processes_done: Event
    ) -> InfidelityWorker:
        return InfidelityWorker(
            queue,
            rank,
            self.world_size,
            all_processes_done,
            self.model_factory,
            self.dataset,
            self.batch_size,
            self.perturbation_generators,
            self.num_perturbations,
            self.activation_fns,
            self._handle_result if self.world_size == 1 else None,
        )
