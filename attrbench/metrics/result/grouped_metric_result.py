from abc import abstractmethod
from attrbench.metrics.result import BatchResult, MetricResult


class GroupedMetricResult(MetricResult):
    """
    Abstract class to represent grouped results of distributed metrics.
    These are results of metrics where the attributions dataset is grouped,
    i.e. the metric is computed for all attribution methods at a time.
    """
    
    @abstractmethod
    def add(self, batch_result: BatchResult):
        """
        Add a batch to the result (abstract method).
        """
        self.tree.write_dict(batch_result.indices.detach().cpu().numpy(),
                             batch_result.results,
                             level_order=self.level_order)
