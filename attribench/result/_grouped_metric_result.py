from abc import abstractmethod
from ._grouped_batch_result import GroupedBatchResult
from ._metric_result import MetricResult


class GroupedMetricResult(MetricResult):
    """
    Abstract class to represent grouped results of distributed metrics.
    These are results of metrics where the attributions dataset is grouped,
    i.e. the metric is computed for all attribution methods at a time.
    This is used for metrics  which have a shared computation
    for all attribution methods to save computation time, e.g. Infidelity.
    """

    @abstractmethod
    def add(self, batch_result: GroupedBatchResult):
        """
        Add a batch to the result (abstract method).
        """
        self.tree.write_dict(
            batch_result.indices.detach().cpu().numpy(),
            batch_result.results,
            level_order=self.level_order,
        )
