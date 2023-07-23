from ...data import AttributionsDataset
from ... import MethodFactory, AttributionMethod
from typing import Dict
import torch


def _randomize_parameters(model: torch.nn.Module) -> torch.nn.Module:
    pass  # TODO


def _parameter_randomization_batch(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    
    method_dict_rand: Dict[str, AttributionMethod],
    device: torch.device,
):
    pass  # TODO


def parameter_randomization(
    model: torch.nn.Module,
    attributions_dataset: AttributionsDataset,
    batch_size: int,
    method_factory: MethodFactory,
    device: torch.device = torch.device("cpu"),
):
    randomized_model = _randomize_parameters(model)
    method_dict_rand = method_factory(randomized_model)

    # TODO

