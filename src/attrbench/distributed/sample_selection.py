from attrbench.distributed import DistributedSampler, DistributedComputation, PartialResultMessage, DoneMessage
from tqdm import tqdm
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


def _sample_selection_worker(model_factory: Callable[[], Model], dataset: Dataset, queue: mp.Queue, rank: int,
                             world_size: int, batch_size: int, sufficient_samples: mp.Event,
                             all_processes_done: mp.Event):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    sampler = DistributedSampler(dataset, world_size, rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    device = torch.device(rank)
    model = model_factory()
    model.to(device)

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        output = torch.argmax(model(batch_x), dim=1)
        correct_samples = batch_x[output == batch_y, ...]
        correct_labels = batch_y[output == batch_y]
        result = SamplesResult(correct_samples.cpu().numpy(), correct_labels.cpu().numpy())
        queue.put(PartialResultMessage(rank, result))
        if sufficient_samples.is_set():
            break
    queue.put(DoneMessage(rank))

    all_processes_done.wait()
    dist.destroy_process_group()


class SampleSelection(DistributedComputation):
    def __init__(self, model_factory: Callable[[], Model], dataset: Dataset, path: str,
                 num_samples: int, sample_shape: Tuple, batch_size: int,
                 address: str = "localhost",
                 port: str = "12355", devices: Tuple = None):
        super().__init__(address, port, devices)
        self.model_factory = model_factory
        self.dataset = dataset
        self.path = path
        self.writer = HDF5DatasetWriter(path, (num_samples, *sample_shape))
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.sufficient_samples = self.ctx.Event()
        self.count = 0
        self.prog = None

    def _create_worker(self, queue: mp.Queue, rank: int, all_processes_done: mp.Event):
        args = (self.model_factory, self.dataset, queue, rank, self.world_size, self.batch_size,
                self.sufficient_samples, all_processes_done)
        return self.ctx.Process(target=_sample_selection_worker, args=args)

    def start(self):
        self.prog = tqdm(total=self.num_samples)
        super().start()

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