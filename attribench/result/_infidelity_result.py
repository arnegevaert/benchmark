from typing import Tuple, Optional, List
from typing_extensions import override
import pandas as pd
from ._grouped_metric_result import GroupedMetricResult


class InfidelityResult(GroupedMetricResult):
    """Represents results from running the Infidelity metric.
    """
    def __init__(
        self,
        method_names: List[str],
        perturbation_generators: List[str],
        activation_fns: List[str],
        num_samples: int,
    ):
        """
        Parameters
        ----------
        method_names : List[str]
            Names of attribution methods tested by Infidelity.
        perturbation_generators : List[str]
            Names of perturbation generators used by Infidelity.
        activation_fns : List[str]
            Names of activation functions used by Infidelity.
        num_samples : int
            Number of samples on which Infidelity was run.
        """
        levels = {
            "method": method_names,
            "perturbation_generator": perturbation_generators,
            "activation_fn": activation_fns,
        }
        shape = [num_samples, 1]
        level_order = ["method", "perturbation_generator", "activation_fn"]
        super().__init__(method_names, shape, levels, level_order)

    @classmethod
    @override
    def _load(cls, path: str, format="hdf5") -> "InfidelityResult":
        tree = cls._load_tree(path, format)
        res = InfidelityResult(
            tree.levels["method"],
            tree.levels["perturbation_generator"],
            tree.levels["activation_fn"],
            tree.shape[0],
        )
        res.tree = tree
        return res

    def get_df(
        self,
        perturbation_generator: str,
        activation_fn: str,
        methods: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        """Retrieves a dataframe from the result for the given perturbation
        generator and activation function. The dataframe contains a row for
        each method and a column for each sample. Each value is the
        Infidelity for the given method on the given sample.

        Parameters
        ----------
        perturbation_generator : str
            The perturbation generator to use.
        activation_fn : str
            The activation function to use.
        methods : Optional[List[str]], optional
            The methods to include. If None, includes all methods.
            Defaults to None.

        Returns
        -------
        Tuple[pd.DataFrame, bool]
            Dataframe containing results,
            and boolean indicating if higher is better.
        """
        methods = methods if methods is not None else self.method_names
        df_dict = {}
        for method in methods:
            df_dict[method] = self.tree.get(
                method=method,
                perturbation_generator=perturbation_generator,
                activation_fn=activation_fn,
            ).flatten()
        return pd.DataFrame.from_dict(df_dict), False
