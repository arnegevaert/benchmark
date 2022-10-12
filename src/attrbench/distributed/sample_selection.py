from attrbench.distributed import DistributedSampler, DistributedComputation, PartialResultMessage, DoneMessage
from attrbench.typing import Model
from attrbench.data import HDF5DatasetWriter
from typing import Callable, Tuple
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import torch
from numpy import typing as npt
import torch.distributed as dist


class SamplesResult:
    def __init__(self, samples: npt.NDArray, labels: npt.NDArray):
        self.samples = samples
        self.labels = labels


class SampleSelection(DistributedComputation):
    def __init__(self, model_factory: Callable[[], Model], dataset: Dataset, path: str,
                 num_samples: int, sample_shape: Tuple, batch_size: int,
                 address: str = "localhost",
                 port: str = "12355", devices: Tuple = None):
        self.model_factory = model_factory
        self.dataset = dataset
        self.path = path
        self.writer = HDF5DatasetWriter(path, (num_samples, *sample_shape))
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.done_event = mp.Event()
        self.count = 0
        super().__init__(address, port, devices)

    def _worker(self, queue: mp.Queue, rank: int, done_event: mp.Event = None):
        sampler = DistributedSampler(self.dataset, self.world_size, rank)
        dataloader = DataLoader(self.dataset, sampler=sampler, batch_size=self.batch_size)
        device = torch.device(rank)
        model = self.model_factory()
        model.to(device)

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_x)
            correct_samples = batch_x[output == batch_y, ...]
            correct_labels = batch_y[output == batch_y]
            result = SamplesResult(correct_samples.cpu().numpy(), correct_labels.cpu().numpy())
            queue.put(PartialResultMessage(rank, result))
            if done_event.is_set():
                break
        queue.put(DoneMessage(rank))

    def _worker_setup(self, queue: mp.Queue, rank: int, done_event: mp.Event):
        dist.init_process_group("gloo", rank=rank, world_size=self.world_size)
        self._worker(queue, rank, self.done_event)
        done_event.wait()
        dist.destroy_process_group()

    def _handle_result(self, result: PartialResultMessage[SamplesResult]):
        samples = result.data.samples
        labels = result.data.labels
        if self.count + samples.shape[0] > self.num_samples:
            samples = samples[:self.num_samples - self.count]
            labels = labels[:self.num_samples - self.count]
            self.done_event.set()
        self.writer.write(samples, labels)
        self.count += samples.shape[0]