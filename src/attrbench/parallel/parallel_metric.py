from torch.utils.data.sampler import T_co
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from attrbench.metrics import Metric
from typing import Tuple, Iterator
import torch
from torch.utils.data import Dataset, Sampler, DataLoader


class DoneMessage:
    def __init__(self, rank):
        self.rank = rank

class PartialResultMessage:
    def __init__(self, rank, indices, result, done=False):
        self.rank = rank
        self.indices = indices
        self.result = result
        self.done = done


class ParallelMetric:
    def __init__(self, metric: Metric, dataset: Dataset, address="localhost", port="12355", devices: Tuple[int] = None):
        # Attrbench metric object
        self.metric = metric
        # Address and port for multiprocessing
        self.address = address
        self.port = port
        # Dataset to evaluate metric on
        self.dataset = dataset
        self.devices = devices if devices is not None else list(range(torch.cuda.device_count()))
        self.world_size = len(devices)

    def _compute_metric(self, rank):
        dist.init_process_group("gloo", rank=rank, world_size=self.world_size)

        sampler = DistributedEvalSampler(self.dataset, self.world_size, rank)
        dataloader = DataLoader(self.dataset, sampler=sampler)

        it = iter(dataloader)
        for batch_indices, batch_x, batch_y in dataloader

        dist.destroy_process_group()

    def evaluate(self):
        # Initialize multiproc parameters
        os.environ["MASTER_ADDR"] = self.address
        os.environ["MASTER_PORT"] = self.port
        mp.set_start_method("spawn")

        queue = mp.Queue()  # Will store results from metric
        global_done_event = mp.Event()  # Used for signaling processes that they can terminate
        processes = []  # Used for joining all processes

        # Start all processes
        for rank in range(self.world_size):
            p = mp.Process(target=self._compute_metric, args=(rank,))
            p.start()
            processes.append(p)

        # Gather results
        # This is not busy waiting: queue.get will block until a result is passed
        results = []
        done_processes = [False for _ in processes]
        while not all(done_processes):
            res = queue.get()  # res is a PartialResultMessage or a DoneMessage
            if type(res) == DoneMessage:
                done_processes[res.rank] = True
            else:
                results.append(res)

        # Processes are now allowed to terminate
        global_done_event.set()
        for p in processes:
            p.join()


class DistributedEvalSampler(Sampler):
    def __init__(self, dataset: Dataset, world_size: int, rank: int):
        """
        DistributedEvalSampler is an alternative to DistributedSampler that
        does not add extra samples to make the total count evenly divisible.
        SplittingSampler also does not shuffle the data.
        Credit: https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py
        """
        super().__init__(None)
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank

        ds_size = len(dataset)
        self.indices = list(range(ds_size))[self.rank:ds_size:self.world_size]

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.indices)


class IndexDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data, target = self.dataset[item]
        return item, data, target