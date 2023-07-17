from abc import abstractmethod
from attribench.data import IndexDataset
from tqdm import tqdm

from .._message import PartialResultMessage
from .._distributed_computation import DistributedComputation
from ._metric_worker import MetricWorker, WorkerConfig
from attribench.result._metric_result import MetricResult
from typing import Tuple, Optional
from attribench._model_factory import ModelFactory


class Metric(DistributedComputation):
    """Abstract base class for metrics that are computed using multiple processes.
    """
    def __init__(
        self,
        model_factory: ModelFactory,
        dataset: IndexDataset,
        batch_size: int,
        address: str,
        port: str | int,
        devices: Optional[Tuple] = None,
    ):
        super().__init__(address, port, devices)
        self.batch_size = batch_size
        self.dataset = dataset
        self.model_factory = model_factory
        self.prog = None  # TQDM progress bar
        self._result: Optional[MetricResult] = None

    @abstractmethod
    def _create_worker(self, worker_config: WorkerConfig) -> MetricWorker:
        raise NotImplementedError

    def _cleanup(self):
        if self.prog is not None:
            self.prog.close()

    def run(self, result_path: Optional[str] = None, progress=True):
        """
        Runs the metric computation and optionally saves the result.
        If no result path is given, the result will not be saved to disk.
        It can still be accessed via the ``result`` property.

        Parameters
        ----------
        result_path : str, optional
            Path to save the result to. If None, the result is not saved to disk.
        progress : bool, optional
            Whether to show a progress bar. Defaults to True.
        """
        if progress:
            self.prog = tqdm(total=len(self.dataset))
        super().run()
        if result_path is not None:
            self.save_result(result_path)

    def save_result(self, path: str, format="hdf5"):
        """Save the result to disk.

        Parameters
        ----------
        path : str
            Path to save the result to.
        format : str, optional
            Format to save the result in.
            If ``"hdf5"``, the result is saved as an HDF5 file.
            If ``"csv"``, the result is saved as a directory structure containing
            CSV files. Default: ``"hdf5"``.

        Raises
        ------
        ValueError
            If the result is None.
        """
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
