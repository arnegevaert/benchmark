from tqdm import tqdm
from torch import nn
import torch.multiprocessing as mp

from attrbench.data import AttributionsDataset
from attrbench.distributed import PartialResultMessage, DistributedComputation
from attrbench.distributed.metrics import MetricWorker
from attrbench.distributed.metrics.result import MetricResult
from typing import Tuple, Callable, Optional


class DistributedMetric(DistributedComputation):
    def __init__(self, model_factory: Callable[[], nn.Module], dataset: AttributionsDataset, batch_size: int,
                 address="localhost", port="12355", devices: Tuple = None):
        super().__init__(address, port, devices)
        self.batch_size = batch_size
        self.dataset = dataset
        self.model_factory = model_factory
        self.prog = None  # TQDM progress bar
        self._result: Optional[MetricResult] = None

    def _create_worker(self, queue: mp.Queue, rank: int, all_processes_done: mp.Event) -> MetricWorker:
        raise NotImplementedError

    def run(self, result_path: str = None, progress=True):
        if progress:
            self.prog = tqdm()
        super().run()
        if result_path is not None:
            self.save_result(result_path)

    def save_result(self, path: str):
        self._result.save(path)

    def _handle_result(self, result_message: PartialResultMessage):
        self._result.add(result_message.data)
        if self.prog is not None:
            self.prog.update(len(result_message.data.indices))