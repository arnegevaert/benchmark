from attrbench.distributed import DistributedSampler, DistributedComputation, PartialResultMessage, DoneMessage, Worker
from tqdm import tqdm
from attrbench.data import HDF5DatasetWriter
from typing import Callable, Tuple
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import torch
from numpy import typing as npt
from torch import nn


class SamplesResult:
    def __init__(self, samples: npt.NDArray, labels: npt.NDArray):
        self.samples = samples
        self.labels = labels


class SampleSelectionWorker(Worker):
    def __init__(self, result_queue: mp.Queue, rank: int, world_size: int, all_processes_done: mp.Event,
                 sufficient_samples: mp.Event, batch_size: int, dataset: Dataset,
                 model_factory: Callable[[], nn.Module]):
        super().__init__(result_queue, rank, world_size, all_processes_done)
        self.sufficient_samples = sufficient_samples
        self.model_factory = model_factory
        self.dataset = dataset
        self.batch_size = batch_size

    def work(self):
        sampler = DistributedSampler(self.dataset, self.world_size, self.rank)
        dataloader = DataLoader(self.dataset, sampler=sampler, batch_size=self.batch_size,
                                num_workers=4, pin_memory=True)
        device = torch.device(self.rank)
        model = self.model_factory()
        model.to(device)
        it = iter(dataloader)

        for batch_x, batch_y in it:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            with torch.no_grad():
                output = torch.argmax(model(batch_x), dim=1)
            correct_samples = batch_x[output == batch_y, ...]
            correct_labels = batch_y[output == batch_y]
            result = SamplesResult(correct_samples.cpu().numpy(), correct_labels.cpu().numpy())
            self.result_queue.put(PartialResultMessage(self.rank, result))
            if self.sufficient_samples.is_set():
                break
        self.result_queue.put(DoneMessage(self.rank))


class SampleSelection(DistributedComputation):
    def __init__(self, model_factory: Callable[[], nn.Module], dataset: Dataset, writer: HDF5DatasetWriter,
                 num_samples: int, batch_size: int,
                 address: str = "localhost",
                 port: str = "12355", devices: Tuple = None):
        super().__init__(address, port, devices)
        self.model_factory = model_factory
        self.dataset = dataset
        self.writer = writer
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.sufficient_samples = self.ctx.Event()
        self.count = 0
        self.prog = None

    def _create_worker(self, queue: mp.Queue, rank: int, all_processes_done: mp.Event):
        return SampleSelectionWorker(queue, rank, self.world_size, all_processes_done, self.sufficient_samples,
                                     self.batch_size, self.dataset, self.model_factory)

    def run(self):
        self.prog = tqdm(total=self.num_samples)
        super().run()

    def _handle_result(self, result: PartialResultMessage[SamplesResult]):
        samples = result.data.samples
        labels = result.data.labels
        if self.count + samples.shape[0] > self.num_samples:
            samples = samples[:self.num_samples - self.count]
            labels = labels[:self.num_samples - self.count]
            self.sufficient_samples.set()
        self.writer.write(samples, labels)
        self.count += samples.shape[0]
        self.prog.update(samples.shape[0])
