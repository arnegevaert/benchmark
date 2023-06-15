from typing import Union, Tuple, Dict, Optional, List

from torch import nn
from torch import multiprocessing as mp
from multiprocessing.synchronize import Event

from attribench.masking import Masker
from ._deletion_worker import DeletionWorker
from attribench.result import DeletionResult, InsertionResult
from .._metric import Metric
from attribench.data import AttributionsDataset
from attribench import ModelFactory


class Deletion(Metric):
    def __init__(
        self,
        model_factory: ModelFactory,
        dataset: AttributionsDataset,
        batch_size: int,
        maskers: Dict[str, Masker],
        activation_fns: Union[List[str], str],
        mode: str = "morf",
        start: float = 0.0,
        stop: float = 1.0,
        num_steps: int = 100,
        address="localhost",
        port="12355",
        devices: Optional[Tuple] = None,
    ):
        """Compute the Deletion metric for a given `AttributionsDataset` and model
        using multiple processes.

        Deletion is computed by iteratively masking the top (Most Relevant First,
        or MoRF) or bottom (Least Relevant First, or LeRF) features of
        the input samples and computing the confidence of the model on the masked
        samples.

        This results in a curve of confidence vs. number of features masked. The
        area under (or equivalently over) this curve is the Deletion metric.

        `start`, `stop`, and `num_steps` are used to determine the range of features
        to mask. The range is determined by `start` and `stop` as a percentage of
        the total number of features. `num_steps` is the number of steps to take
        between `start` and `stop`.

        The Deletion metric is computed for each masker in `maskers` and for each
        activation function in `activation_fns`. The number of processes is
        determined by the number of devices. If `devices` is None, then all
        available devices are used. Samples are distributed evenly across the
        processes.

        Parameters
        ----------
        model_factory : Callable[[], nn.Module]
            ModelFactory instance or callable that returns a model.
            Used to create a model for each subprocess.
        dataset : AttributionsDataset
            Dataset containing the samples and attributions to compute
            Deletion on.
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
            model_factory, dataset, batch_size, address, port, devices
        )
        self.num_steps = num_steps
        self.stop = stop
        self._start = start
        self.mode = mode
        if isinstance(activation_fns, str):
            activation_fns = [activation_fns]
        self.activation_fns = activation_fns
        self.maskers = maskers
        self._result = DeletionResult(
            dataset.method_names,
            tuple(maskers.keys()),
            tuple(self.activation_fns),
            mode,
            shape=(dataset.num_samples, num_steps),
        )
        self.dataset = dataset

    def _create_worker(
        self, queue: mp.Queue, rank: int, all_processes_done: Event
    ) -> DeletionWorker:
        return DeletionWorker(
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


class Insertion(Deletion):
    """
    Computes the insertion metric for a given model and dataset.
    This is a simple wrapper around the deletion metric.
    The only differences are:
    - Start, stop and mode are swapped (deleting the first x% of the input is
    the same as inserting the last x% of the input)
    - The result is an InsertionResult instead of a DeletionResult
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        dataset: AttributionsDataset,
        batch_size: int,
        maskers: Dict[str, Masker],
        activation_fns: Union[List[str], str],
        mode: str = "morf",
        start: float = 0.0,
        stop: float = 1.0,
        num_steps: int = 100,
        address="localhost",
        port="12355",
        devices: Optional[Tuple] = None,
    ):
        super().__init__(
            model_factory,
            dataset,
            batch_size,
            maskers,
            activation_fns,
            "lerf" if mode == "morf" else "morf",  # Swap mode
            1 - start,  # Swap start
            1 - stop,  # Swap stop
            num_steps,
            address,
            port,
            devices,
        )
        self._result = InsertionResult(
            dataset.method_names,
            tuple(maskers.keys()),
            tuple(self.activation_fns),
            mode,
            shape=(dataset.num_samples, num_steps),
        )
