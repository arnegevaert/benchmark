from typing import Dict, Callable, Union, Tuple
import torch
from torch import nn
from torch import multiprocessing as mp
from torch.utils.data import DataLoader

from attrbench.masking import ImageMasker
from attrbench.metrics.deletion import irof
from attrbench.distributed import Worker, DistributedSampler, DoneMessage, PartialResultMessage
from attrbench.distributed.metrics.result import BatchResult
from attrbench.distributed.metrics.deletion import DistributedDeletion
from attrbench.data import AttributionsDataset

from .deletion import DeletionWorker


class IrofWorker(DeletionWorker):
    def work(self):
        sampler = DistributedSampler(self.dataset, self.world_size, self.rank, shuffle=False)
        dataloader = DataLoader(self.dataset, sampler=sampler, batch_size=self.batch_size, num_workers=4)
        device = torch.device(self.rank)
        model = self.model_factory()
        model.to(device)

        for batch_indices, batch_x, batch_y, batch_attr, method_names in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_result = {}
            for masker_name, masker in self.maskers.items():
                if not isinstance(masker, ImageMasker):
                    raise ValueError("Invalid masker", masker_name)
                batch_result[masker_name] = irof(batch_x, batch_y, model, batch_attr.numpy(), masker,
                                                 self.activation_fns, self.mode, self.start, self.stop,
                                                 self.num_steps)
            self.result_queue.put(
                PartialResultMessage(self.rank, BatchResult(batch_indices, batch_result, method_names))
            )
        self.result_queue.put(DoneMessage(self.rank))


class DistributedIrof(DistributedDeletion):
    def __init__(self, model_factory: Callable[[], nn.Module],
                 dataset: AttributionsDataset, batch_size: int,
                 maskers: Dict[str, ImageMasker], activation_fns: Union[Tuple[str], str], mode: str = "morf",
                 start: float = 0., stop: float = 1., num_steps: int = 100,
                 address="localhost", port="12355", devices: Tuple = None):
        super().__init__(model_factory, dataset, batch_size, maskers, activation_fns, mode,
                         start, stop, num_steps, address, port, devices)

    def _create_worker(self, queue: mp.Queue, rank: int, all_processes_done: mp.Event) -> Worker:
        return IrofWorker(queue, rank, self.world_size, all_processes_done, self.model_factory, self.dataset,
                          self.batch_size, self.maskers, self.activation_fns, self.mode,
                          self._start, self.stop, self.num_steps)