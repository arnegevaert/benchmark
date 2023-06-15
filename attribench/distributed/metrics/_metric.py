from attribench.data import IndexDataset
from tqdm import tqdm
from torch import nn
import torch.multiprocessing as mp

from .._message import PartialResultMessage
from .._distributed_computation import DistributedComputation
from ._metric_worker import MetricWorker
from attribench.result._metric_result import MetricResult
from typing import Tuple, Callable, Optional
from multiprocessing.synchronize import Event
from attribench._model_factory import ModelFactory


class Metric(DistributedComputation):
    def __init__(
        self,
        model_factory: ModelFactory,
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
        self, queue: mp.Queue, rank: int, all_processes_done: Event
    ) -> MetricWorker:
        raise NotImplementedError

    def _cleanup(self):
        if self.prog is not None:
            self.prog.close()

    def run(self, result_path: Optional[str] = None, progress=True):
        """
        Runs the metric computation"""
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
        if self._result is None:
            raise ValueError("Cannot get result: result is None")
        return self._result
