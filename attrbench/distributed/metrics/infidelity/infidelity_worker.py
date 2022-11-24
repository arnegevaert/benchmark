from torch import multiprocessing as mp
import torch
import numpy as np
from typing import Callable, Dict, Tuple
from torch import nn
from attrbench.data import AttributionsDataset
from attrbench.util.util import ACTIVATION_FNS
from attrbench.masking import Masker
from attrbench.distributed.metrics import MetricWorker
from attrbench.distributed.metrics.result import BatchResult
from attrbench.distributed import PartialResultMessage, DoneMessage
from attrbench.metrics.infidelity import PerturbationGenerator


class InfidelityWorker(MetricWorker):
    def __init__(self, result_queue: mp.Queue, rank: int, world_size: int, all_processes_done: mp.Event,
                 model_factory: Callable[[], nn.Module], dataset: AttributionsDataset, batch_size: int,
                 perturbation_generators: Dict[str, PerturbationGenerator], num_perturbations: int,
                 activation_fns: Tuple[str]):
        super().__init__(result_queue, rank, world_size, all_processes_done, model_factory, dataset, batch_size)
        self.activation_fns = activation_fns
        self.num_perturbations = num_perturbations
        self.perturbation_generators = perturbation_generators

    def work(self):
        model = self._get_model()

        for batch_indices, batch_x, batch_y, batch_attr in self.dataloader:
            # TODO batch_attr is a dictionary containing all attributions for the given sample
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Attributions need to be moved to the device, because we will be computing dot products between
            # attributions and perturbations later.
            # They also need to have the same shape as the samples for this.
            # Any axis that has length 1 in the attributions is repeated to match the sample shape
            batch_attr = batch_attr.to_numpy()
            for axis in range(len(batch_attr.shape)):
                if batch_attr.shape[axis] == 1:
                    batch_attr = np.repeat(batch_attr, batch_x.shape[axis], axis=axis)
            batch_attr = torch.tensor(batch_attr, device=self.device).flatten(1)

            # Get original model output on the samples (dict: activation_fn -> torch.Tensor)
            orig_output = {}
            with torch.no_grad():
                for fn in self.activation_fns:
                    # [batch_size, 1]
                    orig_output[fn] = ACTIVATION_FNS[fn](model(batch_x)).gather(dim=1, index=batch_y.unsqueeze(-1))

            for pert_name, pert_generator in self.perturbation_generators.items():
                pert_generator.set_samples(batch_x)
                dot_products = []
                pred_diffs = {afn: [] for afn in self.activation_fns}

                for pert_index in range(self.num_perturbations):
                    # Get perturbation vector I and perturbed samples (x - I)
                    perturbation_vector = pert_generator.generate_perturbation()
                    perturbed_x = batch_x - perturbation_vector

                    # Get output of model on perturbed sample
                    with torch.no_grad():
                        perturbed_output = model(perturbed_x)

                    # Save the prediction difference and perturbation vector
                    for fn in self.activation_fns:
                        activated_perturbed_x = ACTIVATION_FNS[fn](perturbed_output)\
                            .gather(dim=1, index=batch_y.unsqueeze(-1))
                        pred_diffs[fn].append(torch.squeeze(orig_output[fn] - activated_perturbed_x))

                    # TODO finish this