from abc import abstractmethod
from .._distributed_sampler import DistributedSampler
from .._worker import Worker, WorkerConfig
from .._message import PartialResultMessage
from attribench.result._batch_result import BatchResult
from attribench.result._grouped_batch_result import GroupedBatchResult
from typing import Callable, Dict
from attribench.data.attributions_dataset._attributions_dataset import (
    GroupedAttributionsDataset,
    AttributionsDataset,
)
from attribench.data import IndexDataset
from torch import nn
from torch.utils.data import DataLoader
import torch


class MetricWorker(Worker):
    def __init__(
        self,
        worker_config: WorkerConfig,
        model_factory: Callable[[], nn.Module],
        dataset: IndexDataset,
        batch_size: int,
    ):
        super().__init__(worker_config)
        self.batch_size = batch_size
        self.dataset = dataset
        self.model_factory = model_factory

        sampler = DistributedSampler(
            self.dataset,
            self.worker_config.world_size,
            self.worker_config.rank,
            shuffle=False,
        )
        self.dataloader = DataLoader(
            self.dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=4,
        )
        self.device = torch.device(self.worker_config.rank)

    def _get_model(self) -> nn.Module:
        model = self.model_factory()
        model.to(self.device)
        return model

    def setup(self):
        self.model = self._get_model()

    def work(self):
        self.setup()

        for (
            batch_indices,
            batch_x,
            batch_y,
            batch_attr,
            method_names,
        ) in self.dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            batch_result = self.process_batch(
                batch_x,
                batch_y,
                batch_attr,
            )
            self.worker_config.send_result(
                PartialResultMessage(
                    self.worker_config.rank,
                    BatchResult(batch_indices, batch_result, method_names),
                )
            )

    @abstractmethod
    def process_batch(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_attr: torch.Tensor,
    ):
        raise NotImplementedError


class GroupedMetricWorker(MetricWorker):
    def __init__(
        self,
        worker_config: WorkerConfig,
        model_factory: Callable[[], nn.Module],
        dataset: GroupedAttributionsDataset,
        batch_size: int,
    ):
        super().__init__(
            worker_config,
            model_factory,
            dataset,
            batch_size,
        )

    def work(self):
        self.setup()

        for (
            batch_indices,
            batch_x,
            batch_y,
            batch_attr,
        ) in self.dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            batch_result = self.process_batch(
                batch_x,
                batch_y,
                batch_attr,
            )
            self.worker_config.send_result(
                PartialResultMessage(
                    self.worker_config.rank,
                    GroupedBatchResult(batch_indices, batch_result),
                )
            )

    @abstractmethod
    def process_batch(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_attr: Dict[str, torch.Tensor],
    ):
        raise NotImplementedError
