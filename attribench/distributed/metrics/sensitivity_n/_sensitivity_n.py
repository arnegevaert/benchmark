from .._metric import Metric
from typing import Dict, Union, Tuple, List
from attribench.data.attributions_dataset._attributions_dataset import (
    AttributionsDataset,
    GroupedAttributionsDataset,
)
from attribench.masking import Masker
from attribench.result._sensitivity_n_result import SensitivityNResult
from ._sensitivity_n_worker import SensitivityNWorker
from ..._worker import WorkerConfig
from attribench._model_factory import ModelFactory


class SensitivityN(Metric):
    """Compute the Sensitivity-n metric for a given :class:`~attribench.data.AttributionsDataset` and model
    using multiple processes.

    Sensitivity-n is computed by iteratively masking a random subset of `n` features
    of the input samples and computing the output of the model on the masked
    samples.

    For each random subset of masked features, the sum of the attributions is
    also computed. This results in two series of values: the model output and
    the sum of the attributions. The Sensitivity-n metric is the correlation
    between these two series.

    This is repeated for different values of `n` between `min_subset_size` and
    `max_subset_size` in `num_steps` steps. `min_subset_size` and `max_subset_size`
    are percentages of the total number of features.
    For each value of `n`, `num_subsets` random subsets are generated.

    If segmented is True, then the Seg-Sensitivity-n metric is computed.
    This metric is analogous to Sensitivity-n, but instead of using random
    subsets of features, the images are first segmented into superpixels and
    then random subsets of superpixels are masked. This improves the
    signal-to-noise ratio of the metric for high-resolution images.

    The Sensitivity-n metric is computed for each masker in `maskers` and for each
    activation function in `activation_fns`. The number of processes is
    determined by the number of devices. If `devices` is None, then all
    available devices are used. Samples are distributed evenly across the
    processes.
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        attributions_dataset: AttributionsDataset,
        batch_size: int,
        maskers: Dict[str, Masker],
        activation_fns: Union[List[str], str],
        min_subset_size: float,
        max_subset_size: float,
        num_steps: int,
        num_subsets: int,
        segmented=False,
        address="localhost",
        port="12355",
        devices: Tuple | None = None,
    ):
        """
        Parameters
        ----------
        model_factory : ModelFactory
            ModelFactory instance or callable that returns a model.
            Used to create a model for each subprocess.
        attributions_dataset : AttributionsDataset
            Dataset containing the attributions to compute Sensitivity-n on.
        batch_size : int
            Batch size to use when computing the model output.
        maskers : Dict[str, Masker]
            Dictionary of maskers to use. Keys are the names of the maskers.
        activation_fns : Union[Tuple[str], str]
            Activation functions to use. If a single string is passed, then the
            it is converted to a single-element list.
        min_subset_size : float
            Minimum percentage of features to mask.
        max_subset_size : float
            Maximum percentage of features to mask.
        num_steps : int
            Number of steps between `min_subset_size` and `max_subset_size`.
        num_subsets : int
            Number of random subsets to generate for each value of `n`.
        segmented : bool
            If True, then the Seg-Sensitivity-n metric is computed.
        address : str, optional
            Address to use for the distributed computation.
            Defaults to "localhost".
        port : str | int, optional
            Port to use for the distributed computation.
            Defaults to "12355".
        devices : Tuple | None
            Devices to use for the distributed computation. If None, then all
            available devices are used.
        """
        super().__init__(
            model_factory,
            attributions_dataset,
            batch_size,
            address,
            port,
            devices,
        )
        if isinstance(activation_fns, str):
            activation_fns = [activation_fns]
        self.activation_fns: List[str] = activation_fns
        self.dataset = GroupedAttributionsDataset(attributions_dataset)
        self.maskers = maskers
        self.num_subsets = num_subsets
        self.num_steps = num_steps
        self.max_subset_size = max_subset_size
        self.min_subset_size = min_subset_size
        self.segmented = segmented
        self._result = SensitivityNResult(
            attributions_dataset.method_names,
            list(maskers.keys()),
            self.activation_fns,
            attributions_dataset.num_samples,
            num_steps,
        )

    def _create_worker(
        self, worker_config: WorkerConfig
    ) -> SensitivityNWorker:
        return SensitivityNWorker(
            worker_config,
            self.model_factory,
            self.dataset,
            self.batch_size,
            self.min_subset_size,
            self.max_subset_size,
            self.num_steps,
            self.num_subsets,
            self.maskers,
            self.activation_fns,
            self.segmented,
        )
