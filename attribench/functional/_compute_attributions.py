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

    TODO don't write to file, just return the dict

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

    result_dict: Dict[str, List[torch.Tensor]] = {method_name: [
        torch.zeros(1) for _ in range(len(index_dataset))
    ] for method_name in method_dict.keys()}
    for batch_indices, batch_x, batch_y in tqdm(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        for method_name, method in method_dict.items():
            with torch.no_grad():
                attrs = method(batch_x, batch_y)
                if writer is None:
                    for idx in batch_indices:
                        result_dict[method_name][idx] = attrs[idx, ...].cpu()
                else:
                    writer.write(
                        batch_indices.cpu().numpy(),
                        attrs.cpu().numpy(),
                        method_name,
                    )
    if writer is None:
        result_dict_cat = {
            method_name: torch.cat(attrs_list)
            for method_name, attrs_list in result_dict.items()
        }
        return result_dict_cat
