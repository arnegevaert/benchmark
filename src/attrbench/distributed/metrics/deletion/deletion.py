from typing import Callable, Union, Tuple, Dict, List

import torch
from torch import nn
from torch import multiprocessing as mp
from torch.utils.data import DataLoader

from attrbench.lib.masking import Masker
from attrbench.distributed import Worker, DistributedSampler, DoneMessage
from attrbench.metrics.deletion import deletion
from attrbench.data import AttributionsDataset


class DeletionResultMessage:
    def __init__(self, indices: torch.Tensor, results: Dict[str, Dict[str, torch.Tensor]], method_names: List[str]):
        self.method_names = method_names
        self.results = results
        self.indices = indices


class DeletionWorker(Worker):
    def __init__(self, result_queue: mp.Queue, rank: int, world_size: int, all_processes_done: mp.Event,
                 model_factory: Callable[[], nn.Module], dataset: AttributionsDataset, batch_size: int,
                 maskers: Dict[str, Masker], activation_fns: Union[Tuple[str], str], mode: str = "morf",
                 start: float = 0., stop: float = 1., num_steps: int = 100):
        super().__init__(result_queue, rank, world_size, all_processes_done)
        self.model_factory = model_factory
        self.dataset = dataset
        self.batch_size = batch_size

        self.maskers = maskers
        self.activation_fns = [activation_fns] if type(activation_fns) == str else list(activation_fns)
        self.mode = mode
        self.start = start
        self.stop = stop
        self.num_steps = num_steps

    def work(self):
        sampler = DistributedSampler(self.dataset, self.world_size, self.rank, shuffle=False)
        dataloader = DataLoader(self.dataset, sampler=sampler, batch_size=self.batch_size)
        device = torch.device(self.rank)
        model = self.model_factory()
        model.to(device)

        for batch_indices, batch_x, batch_y, batch_attr, method_names in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_result = {}
            for masker_name, masker in self.maskers.items():
                batch_result[masker_name] = deletion(batch_x, batch_y, model, batch_attr.to_numpy(), masker,
                                                     self.activation_fns, self.mode, self.start, self.stop,
                                                     self.num_steps)
            self.result_queue.put(DeletionResultMessage(batch_indices, batch_result, method_names))
        self.result_queue.put(DoneMessage(self.rank))