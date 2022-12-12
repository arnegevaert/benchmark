from attrbench.distributed import PartialResultMessage, DoneMessage,  DistributedComputation, DistributedSampler, Worker
from attrbench.data import AttributionsDatasetWriter, IndexDataset
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Tuple, Dict, NewType
import torch
from torch import nn
from numpy import typing as npt
from tqdm import tqdm


AttributionMethod = NewType("AttributionMethod", Callable[[torch.Tensor, torch.Tensor], torch.Tensor])


class AttributionResult:
    def __init__(self, indices: npt.NDArray, attributions: npt.NDArray, method_name: str):
        self.indices = indices
        self.attributions = attributions
        self.method_name = method_name


class AttributionsWorker(Worker):
    def __init__(self, result_queue: mp.Queue, rank: int, world_size: int, all_processes_done: mp.Event,
                 model_factory: Callable[[], nn.Module],
                 method_factory: Callable[[nn.Module], Dict[str, Tuple[AttributionMethod, bool]]],
                 dataset: IndexDataset, batch_size: int):
        super().__init__(result_queue, rank, world_size, all_processes_done)
        self.batch_size = batch_size
        self.dataset = dataset
        self.method_factory = method_factory
        self.model_factory = model_factory

    def work(self):
        sampler = DistributedSampler(self.dataset, self.world_size, self.rank, shuffle=False)
        dataloader = DataLoader(self.dataset, sampler=sampler, batch_size=self.batch_size, num_workers=4)
        device = torch.device(self.rank)
        model = self.model_factory()
        model.to(device)
        method_dict = self.method_factory(model)

        for batch_indices, batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            for method_name, method in method_dict.items():
                with torch.no_grad():
                    attrs = method(batch_x, batch_y)
                    result = AttributionResult(batch_indices.cpu().numpy(), attrs.cpu().numpy(), method_name)
                    self.result_queue.put(PartialResultMessage(self.rank, result))
        self.result_queue.put(DoneMessage(self.rank))


class AttributionsComputation(DistributedComputation):
    def __init__(self, model_factory: Callable[[], nn.Module],
                 method_factory: Callable[[nn.Module], Dict[str, AttributionMethod]],
                 dataset: Dataset, batch_size: int,
                 writer: AttributionsDatasetWriter, address="localhost", port="12355", devices: Tuple = None):
        super().__init__(address, port, devices)
        self.model_factory = model_factory
        self.method_factory = method_factory
        self.dataset = IndexDataset(dataset)
        self.batch_size = batch_size
        self.writer = writer
        self.prog = None

    def run(self):
        self.prog = tqdm()
        super().run()

    def _create_worker(self, queue: mp.Queue, rank: int, all_processes_done: mp.Event) -> Worker:
        return AttributionsWorker(queue, rank, self.world_size, all_processes_done, self.model_factory,
                                  self.method_factory, self.dataset, self.batch_size)

    def _handle_result(self, result_message: PartialResultMessage[AttributionResult]):
        indices = result_message.data.indices
        attributions = result_message.data.attributions
        method_name = result_message.data.method_name
        self.writer.write(indices, attributions, method_name)
        self.prog.update(len(indices))