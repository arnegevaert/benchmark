from torch import multiprocessing as mp
import torch
from typing import Callable, Dict, Tuple, Optional, NoReturn
from torch import nn
from attribench.data import AttributionsDataset
from attribench.activation_fns import ACTIVATION_FNS
from attribench.metrics import MetricWorker
from attribench.metrics.result import BatchResult
from attribench.distributed import PartialResultMessage
from .perturbation_generator import PerturbationGenerator


class InfidelityWorker(MetricWorker):
    def __init__(
        self,
        result_queue: mp.Queue,
        rank: int,
        world_size: int,
        all_processes_done: mp.Event,
        model_factory: Callable[[], nn.Module],
        dataset: AttributionsDataset,
        batch_size: int,
        perturbation_generators: Dict[str, PerturbationGenerator],
        num_perturbations: int,
        activation_fns: Tuple[str],
        result_handler: Optional[
            Callable[[PartialResultMessage], NoReturn]
        ] = None,
    ):
        super().__init__(
            result_queue,
            rank,
            world_size,
            all_processes_done,
            model_factory,
            dataset,
            batch_size,
            result_handler,
        )
        self.activation_fns = activation_fns
        self.num_perturbations = num_perturbations
        self.perturbation_generators = perturbation_generators

    def work(self):
        model = self._get_model()

        for batch_indices, batch_x, batch_y, batch_attr in self.dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # method_name -> perturbation_generator
            # -> activation_fn -> [batch_size, 1]
            batch_result: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {
                method_name: {
                    perturbation_generator: {
                        activation_fn: None
                        for activation_fn in self.activation_fns
                    }
                    for perturbation_generator in self.perturbation_generators.keys()
                }
                for method_name in batch_attr.keys()
            }

            # Attributions need to be moved to the device,
            # because we will be computing dot products between
            # attributions and perturbations later.
            # They also need to have the same shape as the samples for this.
            # Any axis that has length 1 in the attributions is repeated to
            # match the sample shape
            tensor_attributions: Dict[str, torch.Tensor] = {}
            for attribution_method, attributions in batch_attr.items():
                for axis in range(len(attributions.shape)):
                    if attributions.shape[axis] == 1:
                        attributions = torch.repeat_interleave(
                            attributions, batch_x.shape[axis], dim=axis
                        )
                tensor_attributions[attribution_method] = attributions.flatten(
                    1
                ).to(self.device)

            # Get original model output on the samples
            # (dict: activation_fn -> torch.Tensor)
            orig_output = {}
            with torch.no_grad():
                for fn in self.activation_fns:
                    # [batch_size, 1]
                    orig_output[fn] = ACTIVATION_FNS[fn](
                        model(batch_x)
                    ).gather(dim=1, index=batch_y.unsqueeze(-1))

            for (
                pert_name,
                pert_generator,
            ) in self.perturbation_generators.items():
                pert_generator.set_samples(batch_x)
                dot_products: Dict[str, torch.Tensor] = {
                    method: [] for method in batch_attr.keys()
                }
                pred_diffs = {afn: [] for afn in self.activation_fns}

                for pert_index in range(self.num_perturbations):
                    # Get perturbation vector I and perturbed samples (x - I)
                    perturbation_vector = (
                        pert_generator.generate_perturbation()
                    )
                    perturbed_x = batch_x - perturbation_vector

                    # Get output of model on perturbed sample
                    with torch.no_grad():
                        perturbed_output = model(perturbed_x)

                    # Save the prediction difference and perturbation vector
                    for fn in self.activation_fns:
                        activated_perturbed_output = ACTIVATION_FNS[fn](
                            perturbed_output
                        ).gather(dim=1, index=batch_y.unsqueeze(-1))
                        pred_diffs[fn].append(
                            torch.squeeze(
                                orig_output[fn] - activated_perturbed_output
                            )
                        )

                    # Compute dot products of perturbation vectors with all
                    # attributions for each sample
                    for (
                        attribution_method,
                        attributions,
                    ) in tensor_attributions.items():
                        # (batch_size)
                        dot_products[attribution_method].append(
                            (
                                perturbation_vector.flatten(1) * attributions
                            ).sum(dim=-1)
                        )

                # For each method and activation function, compute infidelity
                # activation_fn -> [num_perturbations, batch_size]
                tensor_pred_diffs = {
                    afn: torch.stack(pred_diffs[afn], dim=0)
                    for afn in pred_diffs.keys()
                }
                for method in tensor_attributions.keys():
                    # Dot prodcts for this method
                    method_dot_products = torch.stack(
                        dot_products[method]
                    )  # [num_perturbations, batch_size]
                    # Denominator for normalizing constant beta
                    beta_denominator = torch.mean(
                        method_dot_products**2, dim=0, keepdim=True
                    )  # [1, batch_size]
                    for afn in self.activation_fns:
                        # Numerator for normalizing constant beta depends on
                        # activation function
                        # [1, batch_size]
                        beta_numerator = torch.mean(
                            method_dot_products * tensor_pred_diffs[afn],
                            dim=0,
                            keepdim=True,
                        )
                        beta = beta_numerator / beta_denominator
                        # If attribution map is constant 0,
                        # dot products will be 0 and beta will be nan. Set to 0.
                        beta[torch.isnan(beta)] = 0
                        # [batch_size, 1]
                        infidelity = torch.mean(
                            (
                                beta * method_dot_products
                                - tensor_pred_diffs[afn]
                            )
                            ** 2,
                            dim=0,
                        ).unsqueeze(-1)
                        batch_result[method][pert_name][afn] = (
                            infidelity.cpu().detach().numpy()
                        )
            # Return batch result
            self.send_result(
                PartialResultMessage(
                    self.rank, BatchResult(batch_indices, batch_result)
                )
            )
