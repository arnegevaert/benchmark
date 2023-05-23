from attrbench.distributed import Worker, PartialResultMessage
from torch import multiprocessing as mp
from attrbench.distributed import DistributedSampler
from typing import Callable, Optional, NoReturn
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch


class MetricWorker(Worker):
    def __init__(
        self,
        result_queue: mp.Queue,
        rank: int,
        world_size: int,
        all_processes_done: mp.Event,
        model_factory: Callable[[], nn.Module],
        dataset: Dataset,
        batch_size: int,
        result_handler: Optional[
            Callable[[PartialResultMessage], NoReturn]
        ] = None,
    ):
        super().__init__(
            result_queue,
            rank,
            world_size,
            all_processes_done,
            result_handler,
        )
        self.batch_size = batch_size
        self.dataset = dataset
        self.model_factory = model_factory

        sampler = DistributedSampler(
            self.dataset, self.world_size, self.rank, shuffle=False
        )
        self.dataloader = DataLoader(
            self.dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=4,
        )
        self.device = torch.device(self.rank)

    def _get_model(self) -> nn.Module:
        """
        Produces a model instance from the model_factory
        and sends it to the correct device
        :return: The model
        """
        model = self.model_factory()
        model.to(self.device)
        return model

    def work(self):
        raise NotImplementedError
