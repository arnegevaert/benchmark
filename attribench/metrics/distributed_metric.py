from attribench.data import IndexDataset
from tqdm import tqdm
from torch import nn
import torch.multiprocessing as mp

from attribench.distributed import PartialResultMessage, DistributedComputation
from attribench.metrics import MetricWorker
from attribench.metrics.result import MetricResult
from typing import Tuple, Callable, Optional


class Metric(DistributedComputation):
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        dataset: IndexDataset,
        batch_size: int,
        address="localhost",
        port="12355",
        devices: Optional[Tuple] = None,
    ):
        super().__init__(address, port, devices)
        self.batch_size = batch_size
        self.dataset = dataset
        self.model_factory = model_factory
        self.prog = None  # TQDM progress bar
        self._result: Optional[MetricResult] = None

    def _create_worker(
        self, queue: mp.Queue, rank: int, all_processes_done: mp.Event
    ) -> MetricWorker:
        raise NotImplementedError

    def _cleanup(self):
        if self.prog is not None:
            self.prog.close()

    def run(self, result_path: Optional[str] = None, progress=True):
        if progress:
            self.prog = tqdm(total=len(self.dataset))
        super().run()
        if result_path is not None:
            self.save_result(result_path)

    def save_result(self, path: str, format="hdf5"):
        if self._result is not None:
            self._result.save(path, format)
        else:
            raise ValueError("Cannot save result: result is None")

    def _handle_result(self, result_message: PartialResultMessage):
        if self._result is not None:
            self._result.add(result_message.data)
        if self.prog is not None:
            self.prog.update(len(result_message.data.indices))

    @property
    def result(self) -> MetricResult:
        return self._result
