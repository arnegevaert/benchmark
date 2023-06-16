from typing import Tuple, Dict, Optional

from torch import multiprocessing as mp

from attribench.masking import Masker
from ._minimal_subset_worker import MinimalSubsetWorker
from attribench.result import MinimalSubsetResult
from .._metric import Metric
from attribench.data import AttributionsDataset
from attribench._model_factory import ModelFactory
from multiprocessing.synchronize import Event


class MinimalSubset(Metric):
    def __init__(
        self,
        model_factory: ModelFactory,
        dataset: AttributionsDataset,
        batch_size: int,
        maskers: Dict[str, Masker],
        mode: str = "deletion",
        num_steps: int = 100,
        address="localhost",
        port="12355",
        devices: Optional[Tuple] = None,
    ):
        """Computes Minimal Subset Deletion or Insertion for a given
        `AttributionsDataset` and model using multiple processes.

        Minimal Subset Deletion or Insertion is computed by iteratively masking
        (Deletion) or revealing (Insertion) the top features of the input samples
        and computing the prediction of the model on the masked samples.

        Minimal Subset Deletion is the minimal number of features that must be
        masked to change the model's prediction from its original prediction.
        Minimal Subset Insertion is the minimal number of features that must be
        revealed to get the model's original prediction.

        The Minimal Subset metric is computed for each masker in `maskers`.
        The number of processes is determined by the number of devices. If `devices`
        is None, then all available devices are used. Samples are distributed evenly
        across the processes.

        Parameters
        ----------
        model_factory : ModelFactory
            ModelFactory instance or callable that returns a model.
            Used to create a model for each subprocess.
        dataset : AttributionsDataset
            Dataset containing the samples and attributions to compute
            the Minimal Subset metric for.
        batch_size : int
            Batch size per subprocess to use when computing the metric.
        maskers : Dict[str, Masker]
            Dictionary mapping masker names to `Masker` objects.
        mode : str, optional
            "deletion" or "insertion", by default "deletion"
        num_steps : int, optional
            Number of steps to use when computing the Minimal Subset metric,
            by default 100. More steps will result in a more accurate metric,
            but will take longer to compute.
        address : str, optional
            Address to use for multiprocessing, by default "localhost"
        port : str, optional
            Port to use for multiprocessing, by default "12355"
        devices : Optional[Tuple], optional
            Tuple of devices to use for multiprocessing, by default None.
            If None, all available devices are used.

        Raises
        ------
        ValueError
            If `mode` is not "deletion" or "insertion".
        """
        super().__init__(
            model_factory, dataset, batch_size, address, port, devices
        )
        self.dataset = dataset
        self.num_steps = num_steps
        if mode not in ["deletion", "insertion"]:
            raise ValueError("Mode must be deletion or insertion. Got:", mode)
        self.mode = mode
        self.maskers = maskers
        self._result = MinimalSubsetResult(
            dataset.method_names,
            tuple(maskers.keys()),
            mode,
            shape=(dataset.num_samples, 1),
        )

    def _create_worker(
        self, queue: mp.Queue, rank: int, all_processes_done: Event
    ) -> MinimalSubsetWorker:
        return MinimalSubsetWorker(
            queue,
            rank,
            self.world_size,
            all_processes_done,
            self.model_factory,
            self.dataset,
            self.batch_size,
            self.maskers,
            self.mode,
            self.num_steps,
            self._handle_result if self.world_size == 1 else None,
        )
