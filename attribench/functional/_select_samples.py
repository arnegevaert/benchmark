from torch import nn
from torch.utils.data import Dataset, DataLoader
from attribench.data import HDF5DatasetWriter
import torch
from typing import Optional, Tuple


def _select_samples_batch(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    model: nn.Module,
    device: torch.device,
):
    """Returns the correctly classified samples and their labels.

    Parameters
    ----------
    batch_x : torch.Tensor
    batch_y : torch.Tensor
    model : nn.Module
    device : torch.device

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The correctly classified samples and their labels.
    """
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    with torch.no_grad():
        output = torch.argmax(model(batch_x), dim=1)
    correct_samples = batch_x[output == batch_y, ...]
    correct_labels = batch_y[output == batch_y]
    return correct_samples, correct_labels


def select_samples(
    model: nn.Module,
    dataset: Dataset,
    num_samples: int,
    batch_size: int,
    writer: Optional[HDF5DatasetWriter] = None,
    device: Optional[torch.device] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Select correctly classified samples from a dataset and optionally
    write them to a HDF5 file. If the `writer` is `None`, the
    samples and labels are simply returned. Otherwise, the samples and
    labels are written to the HDF5 file and `None` is returned.

    TODO this function should just return the samples and labels. Use the
    distributed class to write the samples and labels to a file.

    Parameters
    ----------
    model : nn.Module
        Model to use for classification.
    dataset : Dataset
        Torch Dataset containing the samples and labels.
    writer : HDF5DatasetWriter
        Writer to write the samples and labels to, by default None.
    num_samples : int
        Number of correctly classified samples to select.
    batch_size : int
        Batch size to use for the dataloader.
    device : Optional[torch.device], optional
        Device to use, by default None.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor] | None
        If `writer` is `None`, a tuple containing the correctly classified
        samples and their labels. Otherwise, `None`.
    """
    if device is None:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    samples_count = 0
    all_correct_samples, all_correct_labels = [], []
    for batch_x, batch_y in dataloader:
        correct_samples, correct_labels = _select_samples_batch(
            batch_x, batch_y, model, device
        )
        if len(correct_samples) > 0:
            if writer is None:
                all_correct_samples.append(correct_samples)
                all_correct_labels.append(correct_labels)
            else:
                writer.write(
                    correct_samples.cpu().numpy(), correct_labels.cpu().numpy()
                )
            samples_count += len(correct_samples)
        if samples_count >= num_samples:
            break
    if writer is None:
        return torch.cat(all_correct_samples), torch.cat(all_correct_labels)
