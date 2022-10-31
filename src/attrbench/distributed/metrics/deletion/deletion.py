from typing import Callable, Union, Tuple, Dict, List
from tqdm import tqdm

import torch
from torch import nn
from torch import multiprocessing as mp
from torch.utils.data import DataLoader

from attrbench.lib.masking import Masker
from attrbench.distributed import Worker, DistributedSampler, DoneMessage, DistributedComputation, PartialResultMessage
from attrbench.distributed.metrics.deletion import DeletionBatchResult, DeletionResult
from attrbench.metrics.deletion import deletion
from attrbench.data import AttributionsDataset


class DeletionWorker(Worker):
    def __init__(self, result_queue: mp.Queue, rank: int, world_size: int, all_processes_done: mp.Event,
                 model_factory: Callable[[], nn.Module], dataset: AttributionsDataset, batch_size: int,
                 maskers: Dict[str, Masker], activation_fns: List[str], mode: str = "morf",
                 start: float = 0., stop: float = 1., num_steps: int = 100):
        super().__init__(result_queue, rank, world_size, all_processes_done)
        self.model_factory = model_factory
        self.dataset = dataset
        self.batch_size = batch_size

        self.maskers = maskers
        self.activation_fns = activation_fns
        self.mode = mode
        self.start = start
        self.stop = stop
        self.num_steps = num_steps

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
                batch_result[masker_name] = deletion(batch_x, batch_y, model, batch_attr.numpy(), masker,
                                                     self.activation_fns, self.mode, self.start, self.stop,
                                                     self.num_steps)
            self.result_queue.put(
                PartialResultMessage(self.rank, DeletionBatchResult(batch_indices, batch_result, method_names))
            )
        self.result_queue.put(DoneMessage(self.rank))


class DistributedDeletion(DistributedComputation):
    def __init__(self, model_factory: Callable[[], nn.Module],
                 dataset: AttributionsDataset, batch_size: int,
                 maskers: Dict[str, Masker], activation_fns: Union[Tuple[str], str], mode: str = "morf",
                 start: float = 0., stop: float = 1., num_steps: int = 100,
                 address="localhost", port="12355", devices: Tuple = None):
        super().__init__(address, port, devices)
        self.num_steps = num_steps
        self.stop = stop
        self._start = start
        self.mode = mode
        self.activation_fns = [activation_fns] if isinstance(activation_fns, str) else list(activation_fns)
        self.maskers = maskers
        self.model_factory = model_factory
        self.dataset = dataset
        self.batch_size = batch_size
        self.prog = None
        self._result = DeletionResult(dataset.method_names, list(maskers.keys()),
                                      self.activation_fns, mode, shape=(dataset.num_samples, num_steps))

    def run(self, result_path: str = None):
        self.prog = tqdm()
        super().run()
        if result_path is not None:
            self.save_result(result_path)

    def save_result(self, path: str):
        self._result.save(path)

    def _create_worker(self, queue: mp.Queue, rank: int, all_processes_done: mp.Event) -> Worker:
        return DeletionWorker(queue, rank, self.world_size, all_processes_done, self.model_factory, self.dataset,
                              self.batch_size, self.maskers, self.activation_fns, self.mode,
                              self._start, self.stop, self.num_steps)

    def _handle_result(self, result_message: PartialResultMessage[DeletionBatchResult]):
        self._result.add(result_message.data)
        self.prog.update(len(result_message.data.indices))
