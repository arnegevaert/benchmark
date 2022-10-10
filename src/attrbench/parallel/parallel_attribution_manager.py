from attrbench.parallel import PartialResultMessage, DoneMessage, ParallelEvalSampler, IndexDataset, ParallelComputationManager
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Tuple, NewType
import torch
import h5py
import numpy as np


Model = NewType("Model", Callable[[torch.Tensor], torch.Tensor])
AttributionMethod = NewType("AttributionMethod", Callable[[torch.Tensor, torch.Tensor], torch.Tensor])


class AttributionResult:
    def __init__(self, indices: torch.Tensor, attributions: torch.Tensor):
        self.indices = indices
        self.attributions = attributions


class ParallelAttributionManager(ParallelComputationManager):
    def __init__(self, model_factory: Callable[[], Model], method_factory: Callable[[Model], AttributionMethod], dataset: Dataset, batch_size: int, sample_shape: Tuple,
                 filename: str, method_name: str, address="localhost", port="12355", devices: Tuple[int] = None):
        super().__init__(address, port, devices)
        self.model_factory = model_factory
        self.method_factory = method_factory
        self.dataset = IndexDataset(dataset)
        self.batch_size = batch_size
        self.sample_shape = sample_shape
        self.filename = filename
        self.method_name = method_name

    def _worker(self, queue: mp.Queue, rank: int):
        sampler = ParallelEvalSampler(self.dataset, self.world_size, rank)
        dataloader = DataLoader(self.dataset, sampler=sampler, batch_size=self.batch_size)
        device = torch.device(rank)
        model = self.model_factory()
        model.to(device)
        method = self.method_factory(model)

        for batch_indices, batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            attrs = method(batch_x, batch_y)
            queue.put(PartialResultMessage(rank, AttributionResult(batch_indices.cpu().numpy(), attrs.cpu().numpy())))
        queue.put(DoneMessage(rank))

    def start(self):
        # Allocate space in the HDF5 file
        with h5py.File(self.filename, "a") as fp:
            if self.method_name in fp:
                del fp[self.method_name]
            fp.create_dataset(self.method_name, shape=(len(self.dataset), *self.sample_shape))
        # Start attributions
        super().start()

    def _handle_result(self, result_message: PartialResultMessage[AttributionResult]):
        with h5py.File(self.filename, "a") as fp:
            result = result_message.data
            if result.attributions.shape[1:] != self.sample_shape:
                raise ValueError(f"Shape of attributions does not match: expected {self.sample_shape}, received {result.attributions.shape[1:]}")
            fp[self.method_name][result.indices, ...] = result.attributions
