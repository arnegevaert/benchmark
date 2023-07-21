from attribench.data.attributions_dataset._attributions_dataset import (
    AttributionsDataset,
    GroupedAttributionsDataset
)
from .._metric_worker import MetricWorker
from ..._worker import WorkerConfig
from .._metric import Metric
from typing import Optional, Tuple
from torch.utils.data import Dataset
from attribench import MethodFactory
from attribench.result._max_sensitivity_result import MaxSensitivityResult
from ._max_sensitivity_worker import MaxSensitivityWorker
from attribench._model_factory import ModelFactory


class MaxSensitivity(Metric):
    """Compute the Max-Sensitivity metric for a given `Dataset` and attribution
    methods using multiple processes.

    Max-Sensitivity is computed by adding a small amount of uniform noise
    to the input samples and computing the norm of the difference in attributions
    between the original samples and the noisy samples.
    The maximum norm of difference is then taken as the Max-Sensitivity.

    The idea is that a small amount of noise should not change the attributions
    significantly, so the norm of the difference should be small. If the norm
    is large, then the attributions are not robust to small perturbations in the
    input.

    The number of processes is determined by the number of devices. If `devices`
    is None, then all available devices are used. Samples are distributed evenly
    across the processes. Each subprocess computes the Max-Sensitivity for a
    all attribution methods on a subset of the samples.
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        attributions_dataset: AttributionsDataset,
        batch_size: int,
        method_factory: MethodFactory,
        num_perturbations: int,
        radius: float,
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
        attributions_dataset : Dataset
            Torch Dataset to use for computing the Max-Sensitivity.
        batch_size : int
            The batch size per subprocess to use for computing the Max-Sensitivity.
        method_factory : MethodFactory
            MethodFactory instance or callable that returns a dictionary of
            attribution methods, given a model.
        num_perturbations : int
            The number of perturbations to use for computing the Max-Sensitivity.
        radius : float
            The radius of the uniform noise to add to the input samples.
        address : str, optional
            Address to use for the multiprocessing connection.
            Default: "localhost"
        port : str, optional
            Port to use for the multiprocessing connection.
            Default: "12355"
        devices : Optional[Tuple], optional
            Tuple of devices to use for multiprocessing.
            If `None`, all available devices are used.
        """
        super().__init__(
            model_factory,
            attributions_dataset,
            batch_size,
            address,
            port,
            devices,
        )
        self.method_factory = method_factory
        self.num_perturbations = num_perturbations
        self.radius = radius
        self.dataset = GroupedAttributionsDataset(attributions_dataset)
        self._result = MaxSensitivityResult(
            method_factory.get_method_names(), num_samples=attributions_dataset.num_samples
        )

    def _create_worker(self, worker_config: WorkerConfig) -> MetricWorker:
        return MaxSensitivityWorker(
            worker_config,
            self.model_factory,
            self.dataset,
            self.batch_size,
            self.method_factory,
            self.num_perturbations,
            self.radius,
        )
