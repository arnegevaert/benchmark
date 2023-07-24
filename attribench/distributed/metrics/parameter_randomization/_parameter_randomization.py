from typing import Tuple
from attribench._model_factory import ModelFactory
from attribench.data.attributions_dataset._attributions_dataset import AttributionsDataset, GroupedAttributionsDataset
from attribench.distributed.metrics._metric_worker import WorkerConfig
from .._metric import Metric
from ._parameter_randomization_worker import ParameterRandomizationWorker
from attribench.result import ParameterRandomizationResult
from attribench import MethodFactory


class ParameterRandomization(Metric):
    """
    Computes the Parameter Randomization metric for a given
    :class:`~attribench.data.AttributionsDataset` and model using multiple processes.

    The Parameter Randomization metric is computed by randomly re-initializing the
    parameters of the model and computing an attribution map of the prediction
    on the re-initialized model. The metric value is the spearman rank correlation
    between the original attribution map and the attribution map of the
    re-initialized model. If this value is high, then the attribution method is
    insensitive to the model parameters, thereby failing the sanity check.

    Source: Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I.,
    Hardt, M., & Kim, B. (2018). Sanity checks for saliency maps.
    Advances in neural information processing systems, 31.
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        attributions_dataset: AttributionsDataset,
        batch_size: int,
        method_factory: MethodFactory,
        address="localhost",
        port=12355,
        devices: Tuple | None = None,
    ):
        """
        Parameters
        ----------
        model_factory : ModelFactory
            ModelFactory instance or callable that returns a model.
            Used to create a model for each subprocess, and to create a
            randomized copy of the model.
        attributions_dataset : AttributionsDataset
            Dataset containing the samples and attributions to compute
            the Parameter Randomization metric for.
        batch_size : int
            Batch size per subprocess to use when computing the metric.
        address : str
            Address to use for the distributed computation.
        port : str | int
            Port to use for the distributed computation.
        devices : Tuple | None, optional
            Tuple of devices to use for the distributed computation.
            If None, then all available devices are used.
        """
        super().__init__(
            model_factory,
            attributions_dataset,
            batch_size,
            address,
            port,
            devices,
        )
        self.dataset = GroupedAttributionsDataset(attributions_dataset)
        self._result = ParameterRandomizationResult(
            attributions_dataset.method_names, attributions_dataset.num_samples
        )
        self.method_factory = method_factory
        
        self.agg_fn = None
        self.agg_dim = None
        if attributions_dataset.aggregate_fn is not None:
            self.agg_fn = attributions_dataset.aggregate_fn
            self.agg_dim = attributions_dataset.aggregate_dim

    def _create_worker(
        self, worker_config: WorkerConfig
    ) -> ParameterRandomizationWorker:
        
        return ParameterRandomizationWorker(
            worker_config,
            self.model_factory,
            self.dataset,
            self.batch_size,
            self.method_factory,
            self.agg_fn,
            self.agg_dim,
        )
