import torch.distributed as dist
import torch.multiprocessing as mp
import os

from attrbench.metrics import Metric
from attrbench.distributed import PartialResultMessage, DoneMessage, DistributedComputation
from attrbench.typing import Model
from typing import Tuple, Callable
import torch
from torch.utils.data import Dataset, Sampler, DataLoader


class DistributedMetric(DistributedComputation):
    def __init__(self, model_factory: Callable[[], Model], metric: Metric, dataset: Dataset, address="localhost", port="12355", devices: Tuple[int] = None):
        super().__init__(address, port, devices)
        # Attrbench metric object
        self.metric = metric
        # Dataset to evaluate metric on
        self.dataset = dataset

    def _compute_metric(self, rank):
        dist.init_process_group("gloo", rank=rank, world_size=self.world_size)

        sampler = ParallelEvalSampler(self.dataset, self.world_size, rank)
        dataloader = DataLoader(self.dataset, sampler=sampler)

        it = iter(dataloader)
        for batch_indices, batch_x, batch_y in dataloader:
            pass  # TODO

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

