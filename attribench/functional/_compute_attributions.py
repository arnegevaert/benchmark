from tqdm import tqdm
from typing import Dict, Optional, List
from torch import nn
from torch.utils.data import Dataset, DataLoader
from attribench.data import AttributionsDatasetWriter, IndexDataset
from attribench import AttributionMethod
import torch


def compute_attributions(
    model: nn.Module,
    method_dict: Dict[str, AttributionMethod],
    dataset: Dataset,
    batch_size: int,
    writer: Optional[AttributionsDatasetWriter] = None,
    device: Optional[torch.device] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Compute attributions for a given model and dataset using a dictionary of
    attribution methods, and optionally write them to a HDF5 file. If the `writer`
    is `None`, the attributions are simply returned in a dictionary.
    Otherwise, the attributions are written to the HDF5 file and `None` is returned.

    Parameters
    ----------
    model : nn.Module
        The model for which the attributions should be computed.
    method_dict : Dict[str, AttributionMethod]
        Dictionary of attribution methods.
    dataset : Dataset
        Torch Dataset to use for computing the attributions.
    batch_size : int
        The batch size to use for computing the attributions.
    writer : Optional[AttributionsDatasetWriter], optional
        AttributionsDatasetWriter to write the attributions to, by default `None`.
        If `None`, the attributions are returned in a dictionary.
    device : Optional[torch.device], optional
        Device to use, by default `None`.
        If `None`, the CPU is used.

    Returns
    -------
    Optional[Dict[str, torch.Tensor]]
        If `writer` is `None`, a dictionary of attributions.
    """
    if device is None:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    index_dataset = IndexDataset(dataset)
    dataloader = DataLoader(
        index_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )

    num_samples = len(index_dataset)
    sample_shape = None
    result_dict: Dict[str, torch.Tensor] = {}
    for batch_indices, batch_x, batch_y in tqdm(dataloader):
        if sample_shape is None:
            sample_shape = batch_x.shape[1:]
            result_dict = {
                method_name: torch.zeros(num_samples, *sample_shape)
                for method_name in method_dict.keys()
            }
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        for method_name, method in method_dict.items():
            attrs = method(batch_x, batch_y)
            if writer is None:
                result_dict[method_name][
                    batch_indices, ...
                ] = attrs.detach().cpu()
            else:
                writer.write(
                    batch_indices.detach().cpu().numpy(),
                    attrs.detach().cpu().numpy(),
                    method_name,
                )
    if writer is None:
        return result_dict
