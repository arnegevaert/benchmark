from tqdm import tqdm
from ...data.attributions_dataset._attributions_dataset import (
    AttributionsDataset,
    GroupedAttributionsDataset,
)
from ... import MethodFactory, AttributionMethod, ModelFactory
from ...result import ParameterRandomizationResult
from ...result._grouped_batch_result import GroupedBatchResult
from typing import Dict, Callable
import torch
from torch.utils.data import DataLoader
from ..._stat import rowwise_spearmanr


def _randomize_parameters(model_factory: ModelFactory) -> torch.nn.Module:
    randomized_model = model_factory()
    for layer in randomized_model.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    return randomized_model


def _parameter_randomization_batch(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_attr: Dict[str, torch.Tensor],
    method_dict_rand: Dict[str, AttributionMethod],
    device: torch.device,
    agg_fn: Callable[
        [
            torch.Tensor,
            int,
        ],
        torch.Tensor,
    ]
    | None = None,
    agg_dim: int | None = None,
) -> Dict[str, torch.Tensor]:
    if set(method_dict_rand.keys()) != set(batch_attr.keys()):
        raise ValueError(
            "Method dictionary and batch attributions dictionary"
            " must have the same keys."
        )
    result: Dict[str, torch.Tensor] = {
        method_name: torch.zeros(1) for method_name in method_dict_rand.keys()
    }
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    for method_name, method in method_dict_rand.items():
        # If attributions were aggregated, we need to perform the same
        # aggregations on the randomized attributions
        attrs_rand = method(batch_x, batch_y).detach().cpu()
        if agg_fn is not None:
            assert agg_dim is not None
            # agg_dim is expressed in terms of sample dimension, need to add
            # 1 to account for batch dimension
            attrs_rand = agg_fn(attrs_rand, agg_dim+1)
        attrs_rand = attrs_rand.flatten(1)

        attrs_orig = batch_attr[method_name].cpu().flatten(1)
        result[method_name] = torch.tensor(
            rowwise_spearmanr(attrs_rand.numpy(), attrs_orig.numpy())
        )

    return result


def parameter_randomization(
    model_factory: ModelFactory,
    attributions_dataset: AttributionsDataset,
    batch_size: int,
    method_factory: MethodFactory,
    device: torch.device = torch.device("cpu"),
) -> ParameterRandomizationResult:
    """
    Computes the Parameter Randomization metric for a given
    :class:`~attribench.data.AttributionsDataset`.

    The Parameter Randomization metric is computed by randomly re-initializing the
    parameters of the model and computing an attribution map of the prediction
    on the re-initialized model. The metric value is the spearman rank correlation
    between the original attribution map and the attribution map of the
    re-initialized model. If this value is high, then the attribution method is
    insensitive to the model parameters, thereby failing the sanity check.

    Source: Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I.,
    Hardt, M., & Kim, B. (2018). Sanity checks for saliency maps.
    Advances in neural information processing systems, 31.

    Parameters
    ----------
    model_factory : ModelFactory
        ModelFactory instance or callable that returns a model.
        Used to create the original model and a randomized copy.
    attributions_dataset : AttributionsDataset
        Dataset containing the samples and attributions to compute
        the Parameter Randomization metric for.
    batch_size : int
        Batch size to use when computing the metric.
    method_factory : MethodFactory
        MethodFactory instance or callable that returns a dictionary mapping
        method names to AttributionMethod objects.
    device : torch.device, optional
        Device to use when computing the metric, by default torch.device("cpu")

    Returns
    -------
    ParameterRandomizationResult
        Result of the Parameter Randomization metric computation.
    """
    
    randomized_model = _randomize_parameters(model_factory)
    randomized_model.to(device)
    randomized_model.eval()

    method_dict_rand = method_factory(randomized_model)
    grouped_dataset = GroupedAttributionsDataset(attributions_dataset)
    dataloader = DataLoader(
        grouped_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    method_names = method_factory.get_method_names()
    result = ParameterRandomizationResult(
        method_names, num_samples=len(grouped_dataset)
    )

    for batch_indices, batch_x, batch_y, batch_attr in tqdm(dataloader):
        agg_fn = None
        agg_dim = None
        if attributions_dataset.aggregate_fn is not None:
            agg_fn = attributions_dataset.aggregate_fn
            agg_dim = attributions_dataset.aggregate_dim

        batch_result = _parameter_randomization_batch(
            batch_x,
            batch_y,
            batch_attr,
            method_dict_rand,
            device,
            agg_fn,
            agg_dim,
        )
        result.add(GroupedBatchResult(batch_indices, batch_result))
    return result
