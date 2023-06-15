from typing import Callable, Union, Tuple, List, Optional, Mapping
import torch
from torch import multiprocessing as mp
from torch.utils.data import DataLoader
from attribench.distributed._message import PartialResultMessage

from attribench.masking import ImageMasker, Masker
from attribench._model_factory import ModelFactory
from ..._worker import Worker
from ..._distributed_sampler import DistributedSampler
from ..._message import PartialResultMessage
from attribench.result._batch_result import BatchResult
from ._deletion import Deletion
from ._deletion_worker import DeletionWorker
from attribench.data import AttributionsDataset
from attribench.functional.metrics._irof import irof_batch

from multiprocessing.synchronize import Event


class IrofWorker(DeletionWorker):
    def __init__(
        self,
        result_queue: mp.Queue,
        rank: int,
        world_size: int,
        all_processes_done: Event,
        model_factory: ModelFactory,
        dataset: AttributionsDataset,
        batch_size: int,
        maskers: Mapping[str, ImageMasker],
        activation_fns: List[str],
        mode: str = "morf",
        start: float = 0,
        stop: float = 1,
        num_steps: int = 100,
        result_handler: Callable[[PartialResultMessage], None] | None = None,
    ):
        super().__init__(
            result_queue,
            rank,
            world_size,
            all_processes_done,
            model_factory,
            dataset,
            batch_size,
            maskers,
            activation_fns,
            mode,
            start,
            stop,
            num_steps,
            result_handler,
        )
        self.maskers = maskers

    def work(self):
        sampler = DistributedSampler(
            self.dataset, self.world_size, self.rank, shuffle=False
        )
        dataloader = DataLoader(
            self.dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=4,
        )
        device = torch.device(self.rank)
        model = self.model_factory()
        model.to(device)

        for (
            batch_indices,
            batch_x,
            batch_y,
            batch_attr,
            method_names,
        ) in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_result = irof_batch(
                batch_x,
                batch_y,
                model,
                batch_attr.numpy(),
                self.maskers,
                self.activation_fns,
                self.mode,
                self.start,
                self.stop,
                self.num_steps,
            )
            self.send_result(
                PartialResultMessage(
                    self.rank,
                    BatchResult(batch_indices, batch_result, method_names),
                )
            )


class Irof(Deletion):
    def __init__(
        self,
        model_factory: ModelFactory,
        dataset: AttributionsDataset,
        batch_size: int,
        maskers: Mapping[str, ImageMasker],
        activation_fns: Union[List[str], str],
        mode: str = "morf",
        start: float = 0.0,
        stop: float = 1.0,
        num_steps: int = 100,
        address="localhost",
        port="12355",
        devices: Optional[Tuple] = None,
    ):
        """Compute the IROF metric for a given `AttributionsDataset` and model
        using multiple processes.

        IROF starts segmenting the input image using SLIC. Then, it iteratively
        masks out the top (Most Relevant First, or MoRF) or bottom (Least Relevant
        First, or LeRF) segments and computes the confidence of the model on the
        masked samples. The relevance of a segment is computed as the average
        relevance of the features in the segment.

        This results in a curve of confidence vs. number of segments masked. The
        area under (or equivalently over) this curve is the IROF metric.

        `start`, `stop`, and `num_steps` are used to determine the range of segments
        to mask. The range is determined by `start` and `stop` as a percentage of
        the total number of segments. `num_steps` is the number of steps to take
        between `start` and `stop`.

        The IROF metric is computed for each masker in `maskers` and for each
        activation function in `activation_fns`. The number of processes is
        determined by the number of devices. If `devices` is None, then all
        available devices are used. Samples are distributed evenly across the
        processes.

        Parameters
        ----------
        model_factory : ModelFactory
            ModelFactory instance or callable that returns a model.
            Used to create a model for each subprocess.
        dataset : AttributionsDataset
            Dataset containing the samples and attributions to compute
            IROF on.
        batch_size : int
            The batch size to use when computing the metric.
        maskers : Dict[str, Masker]
            Dictionary of maskers to use for computing the metric.
        activation_fns : Union[List[str], str]
            List of activation functions to use for computing the metric.
            If a single string is given, it is converted to a single-element
            list.
        mode : str, optional
            Mode to use for computing the metric. Either "morf" or "lerf".
            Default: "morf"
        start : float, optional
            Relative start of the range of features to mask. Must be between 0 and 1.
            Default: 0.0
        stop : float, optional
            Relative end of the range of features to mask. Must be between 0 and 1.
            Default: 1.0
        num_steps : int, optional
            Number of steps to use for the range of features to mask.
            Default: 100
        address : str, optional
            Address to use for the multiprocessing connection.
            Default: "localhost"
        port : str, optional
            Port to use for the multiprocessing connection.
            Default: "12355"
        devices : Optional[Tuple], optional
            Devices to use. If None, then all available devices are used.
            Default: None
        """
        super().__init__(
            model_factory,
            dataset,
            batch_size,
            maskers,
            activation_fns,
            mode,
            start,
            stop,
            num_steps,
            address,
            port,
            devices,
        )
        self.maskers = maskers

    def _create_worker(
        self, queue: mp.Queue, rank: int, all_processes_done: Event
    ) -> Worker:
        return IrofWorker(
            queue,
            rank,
            self.world_size,
            all_processes_done,
            self.model_factory,
            self.dataset,
            self.batch_size,
            self.maskers,
            self.activation_fns,
            self.mode,
            self._start,
            self.stop,
            self.num_steps,
            self._handle_result if self.world_size == 1 else None,
        )
