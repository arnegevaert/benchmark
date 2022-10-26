import torch.distributed as dist
import torch.multiprocessing as mp
import os

from attrbench.metrics import Metric
from attrbench.distributed import PartialResultMessage, DoneMessage, DistributedComputation, Worker
from typing import Tuple, Callable
import torch
from torch.utils.data import Dataset, Sampler, DataLoader


class DistributedMetric(DistributedComputation):
    def __init__(self):
        pass

    def _create_worker(self, queue: mp.Queue, rank: int, all_processes_done: mp.Event) -> Worker:
        pass

    def start(self):
        pass

    def _handle_result(self, result: PartialResultMessage):
        pass