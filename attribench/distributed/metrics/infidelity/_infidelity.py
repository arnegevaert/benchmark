from .._metric import Metric
from typing import Dict, Tuple, Optional, List
from attribench.data.attributions_dataset._attributions_dataset import GroupedAttributionsDataset, AttributionsDataset
from ._infidelity_worker import InfidelityWorker
from ..._worker import WorkerConfig
from attribench.result import InfidelityResult
from attribench.functional.metrics.infidelity._perturbation_generator import (
    PerturbationGenerator,
)
from attribench._model_factory import ModelFactory


class Infidelity(Metric):
    """Computes the Infidelity metric for a given :class:`~attribench.data.AttributionsDataset` and
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
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        attributions_dataset: AttributionsDataset,
        batch_size: int,
        activation_fns: List[str],
        perturbation_generators: Dict[str, PerturbationGenerator],
        num_perturbations: int,
        address="localhost",
        port="12355",
        devices: Optional[Tuple] = None,
    ):
        """
        Parameters
        ----------
        model_factory : ModelFactory
            ModelFactory instance or callable that returns a model.
            Used to create a model for each subprocess.
        attributions_dataset : AttributionsDataset
            Dataset containing the samples and attributions to compute
            Infidelity on.
        batch_size : int
            Batch size to use when computing Infidelity.
        activation_fns : Tuple[str]
            Tuple of activation functions to use when computing Infidelity.
        perturbation_generators : Dict[str, PerturbationGenerator]
            Dictionary of perturbation generators to use for generating
            perturbations.
        num_perturbations : int
            Number of perturbations to generate for each sample.
        address : str, optional
            Address to use for the multiprocessing connection,
            by default "localhost"
        port : str, optional
            Port to use for the multiprocessing connection,
            by default "12355"
        devices : Optional[Tuple], optional
            Devices to use. If None, then all available devices are used.
            By default None.
        """
        super().__init__(
            model_factory, attributions_dataset, batch_size, address, port, devices
        )
        self.dataset = GroupedAttributionsDataset(attributions_dataset)
        self.activation_fns = activation_fns
        self.num_perturbations = num_perturbations
        self.perturbation_generators = perturbation_generators
        self._result = InfidelityResult(
            self.dataset.method_names,
            list(self.perturbation_generators.keys()),
            self.activation_fns,
            num_samples=self.dataset.num_samples,
        )

    def _create_worker(self, worker_config: WorkerConfig) -> InfidelityWorker:
        return InfidelityWorker(
            worker_config,
            self.model_factory,
            self.dataset,
            self.batch_size,
            self.perturbation_generators,
            self.num_perturbations,
            self.activation_fns,
        )
