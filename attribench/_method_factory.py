from typing import Dict, Type, Union, Tuple, List
from torch import nn

from attribench._attribution_method import AttributionMethod


ConfigDict = Dict[
        str,
        Union[
            Type[AttributionMethod],
            Tuple[Type[AttributionMethod], Dict],
        ],
    ]
"""
A ConfigDict is a dictionary mapping strings to either an AttributionMethod,
or a tuple consisting of an AttributionMethod constructor and a dictionary of
keyword arguments to pass to the constructor.
"""


class MethodFactory:
    """
    This class accepts a config dictionary for attribution methods in its
    constructor, and will return a dictionary of ready-to-use AttributionMethod
    objects when called with a model (nn.Module) as argument. This allows
    the attribution methods to be instantiated in subprocesses, which is
    necessary for computing attributions on multiple GPUs, as the methods need
    access to the specific copy of the model for their process.

    The config dictionary should map strings to either AttributionMethod
    constructors, or tuples consisting of an AttributionMethod constructor
    and a dictionary of keyword arguments to pass to the constructor.

    Example::

        {
            "method1": AttributionMethod1,
            "method2": (AttributionMethod2, {"kwarg1": 1, "kwarg2": 2}),
        }

    """

    def __init__(self, config_dict: ConfigDict) -> None:
        """
        Parameters
        ----------
        config_dict : ConfigDict
            Dictionary mapping strings to either AttributionMethod constructors,
            or tuples consisting of an AttributionMethod constructor and a
            dictionary of keyword arguments to pass to the constructor.
        """
        self.config_dict = config_dict

    def __call__(self, model: nn.Module) -> Dict[str, AttributionMethod]:
        """Create dictionary mapping method names to AttributionMethod objects.

        Parameters
        ----------
        model : nn.Module
            Model to compute attributions for.

        Returns
        -------
        Dict[str, AttributionMethod]
            Dictionary mapping method names to AttributionMethod objects.
        """
        result: Dict[str, AttributionMethod] = {}
        for method_name, entry in self.config_dict.items():
            if isinstance(entry, Tuple):
                # Entry consists of constructor and kwargs
                constructor, kwargs = entry
                result[method_name] = constructor(model, **kwargs)
            else:
                # Constructor has no kwargs
                result[method_name] = entry(model)
        return result

    def __len__(self) -> int:
        return len(self.config_dict)

    def get_method_names(self) -> List[str]:
        return list(self.config_dict.keys())
