from ._deletion import Deletion
from attribench._model_factory import ModelFactory
from attribench.masking import Masker
from attribench.data import AttributionsDataset
from attribench.result._insertion_result import InsertionResult
from typing import Dict, List, Optional, Tuple, Union


class Insertion(Deletion):
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
        """Compute the Insertion metric for a given `AttributionsDataset` and model
        using multiple processes. Insertion can be viewed as an opposite version
        of the Deletion metric.

        Insertion is computed by iteratively revealing the top (Most Relevant First,
        or MoRF) or bottom (Least Relevant First, or LeRF) features of
        the input samples, leaving the other features masked out,
        and computing the confidence of the model on the masked samples.

        This results in a curve of confidence vs. number of features masked. The
        area under (or equivalently over) this curve is the Insertion metric.

        `start`, `stop`, and `num_steps` are used to determine the range of features
        to mask. The range is determined by `start` and `stop` as a percentage of
        the total number of features. `num_steps` is the number of steps to take
        between `start` and `stop`.

        The Insertion metric is computed for each masker in `maskers` and for each
        activation function in `activation_fns`. The number of processes is
        determined by the number of devices. If `devices` is None, then all
        available devices are used. Samples are distributed evenly across the
        processes.

        Note that the Insertion metric is equivalent to the Deletion metric
        with the following changes:
        - Start and stop are 1 - start and 1 - stop, respectively
        - The mode parameter is swapped

        Note also that, if start and stop are 1 and 0 or vice versa, then
        Insertion-morf and Deletion-lerf are equal, and Insertion-lerf
        and Deletion-morf are equal.


        Parameters
        ----------
        model_factory : ModelFactory
            ModelFactory instance or callable that returns a model.
            Used to create a model for each subprocess.
        dataset : AttributionsDataset
            Dataset containing the samples and attributions to compute
            Insertion on.
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