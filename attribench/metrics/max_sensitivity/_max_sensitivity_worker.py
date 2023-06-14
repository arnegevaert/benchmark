from attribench.data.index_dataset import IndexDataset
import math
from attribench.distributed.message import PartialResultMessage
from attribench.metrics._metric_worker import MetricWorker
from typing import Callable, Dict, Optional, NoReturn
import torch
from torch import nn
from torch import multiprocessing as mp
from attribench.result._batch_result import BatchResult

from attribench._method_factory import MethodFactory


def _normalize_attrs(attrs):
    flattened = attrs.flatten(1)
    return flattened / torch.norm(flattened, dim=1, p=math.inf, keepdim=True)


class MaxSensitivityWorker(MetricWorker):
    def __init__(
        self,
        result_queue: mp.Queue,
        rank: int,
        world_size: int,
        all_processes_done,
        model_factory: Callable[[], nn.Module],
        method_factory: MethodFactory,
        dataset: IndexDataset,
        batch_size: int,
        num_perturbations: int,
        radius: float,
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
        self.method_factory = method_factory
        self.num_perturbations = num_perturbations
        self.radius = radius

    def work(self):
        model = self._get_model()

        # Get method dictionary
        method_dict = self.method_factory(model)

        for batch_indices, batch_x, batch_y in self.dataloader:
            batch_result: Dict[str, torch.Tensor] = {
                method_name: None for method_name in method_dict.keys()
            }

            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Compute Max-Sensitivity for each method
            for method_name, method in method_dict.items():
                attrs = _normalize_attrs(
                    method(batch_x, batch_y).detach()
                ).cpu()
                diffs = []

                for _ in range(self.num_perturbations):
                    # Add uniform noise with infinity norm <= radius
                    # torch.rand generates noise between 0 and 1
                    # => This generates noise between -radius and radius
                    noise = (
                        torch.rand(batch_x.shape, device=self.device)
                        * 2
                        * self.radius
                        - self.radius
                    )
                    noisy_samples = batch_x + noise
                    # Get new attributions from noisy samples
                    noisy_attrs = _normalize_attrs(
                        method(noisy_samples, batch_y).detach()
                    )
                    # Get relative norm of attribution difference
                    # [batch_size]
                    diffs.append(
                        torch.norm(noisy_attrs.cpu() - attrs, dim=1).detach()
                    )
                # [batch_size, num_perturbations]
                diffs = torch.stack(diffs, 1)
                # [batch_size]
                batch_result[method_name] = diffs.max(dim=1)[0].cpu()
            self.send_result(
                PartialResultMessage(
                    self.rank, BatchResult(batch_indices, batch_result)
                )
            )
