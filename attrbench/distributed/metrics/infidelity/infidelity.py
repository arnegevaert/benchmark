from attrbench.distributed.metrics import DistributedMetric
import warnings
from torch import nn
from typing import Callable, Dict, Tuple
from attrbench.data import AttributionsDataset
from attrbench.distributed.metrics.infidelity import InfidelityWorker, InfidelityResult
from attrbench.metrics.infidelity import PerturbationGenerator
from torch import multiprocessing as mp


class DistributedInfidelity(DistributedMetric):
    def __init__(self, model_factory: Callable[[], nn.Module], dataset: AttributionsDataset, batch_size: int,
                 perturbation_generators: Dict[str, PerturbationGenerator], num_perturbations: int,
                 activation_fns: Tuple[str]):
        super().__init__(model_factory, dataset, batch_size)
        if not dataset.group_attributions:
            warnings.warn("Infidelity expects a dataset group_attributions==True. Setting to True.")
            dataset.group_attributions = True
        self.activation_fns = activation_fns
        self.num_perturbations = num_perturbations
        self.perturbation_generators = perturbation_generators
        self._result = InfidelityResult(self.dataset.method_names, tuple(self.perturbation_generators.keys()),
                                        self.activation_fns, shape=(self.dataset.num_samples, 1))

    def _create_worker(self, queue: mp.Queue, rank: int, all_processes_done: mp.Event) -> InfidelityWorker:
        return InfidelityWorker(queue, rank, self.world_size, all_processes_done, self.model_factory,
                                self.dataset, self.batch_size, self.perturbation_generators,
                                self.num_perturbations, self.activation_fns)
