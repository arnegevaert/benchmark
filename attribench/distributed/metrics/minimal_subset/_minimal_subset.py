from typing import Tuple, Dict, Optional
from ..._worker import WorkerConfig
from attribench.masking import Masker
from ._minimal_subset_worker import MinimalSubsetWorker
from attribench.result import MinimalSubsetResult
from .._metric import Metric
from attribench.data import AttributionsDataset
from attribench._model_factory import ModelFactory


class MinimalSubset(Metric):
    """Computes Minimal Subset Deletion or Insertion for a given
    :class:`~attribench.data.AttributionsDataset` and model using multiple processes.

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
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        attributions_dataset: AttributionsDataset,
        batch_size: int,
        maskers: Dict[str, Masker],
        mode: str = "deletion",
        num_steps: int = 100,
        address="localhost",
        port="12355",
        devices: Optional[Tuple] = None,
    ):
        """
        Parameters
        ----------
        model_factory : ModelFactory
            ModelFactory instance or callable that returns a model.
            Used to create a model for each subprocess.
        attributions_dataset : AttributionsDataset
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
            model_factory, attributions_dataset, batch_size, address, port, devices
        )
        self.dataset = attributions_dataset
        self.num_steps = num_steps
        if mode not in ["deletion", "insertion"]:
            raise ValueError("Mode must be deletion or insertion. Got:", mode)
        self.mode = mode
        self.maskers = maskers
        self._result = MinimalSubsetResult(
            attributions_dataset.method_names,
            list(maskers.keys()),
            mode,
            num_samples=attributions_dataset.num_samples,
        )

    def _create_worker(
        self, worker_config: WorkerConfig
    ) -> MinimalSubsetWorker:
        return MinimalSubsetWorker(
            worker_config,
            self.model_factory,
            self.dataset,
            self.batch_size,
            self.maskers,
            self.mode,
            self.num_steps,
        )
