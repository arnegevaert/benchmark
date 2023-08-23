import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Dict


def _plot_heatmap(
    fig: Figure,
    ax: plt.Axes,
    attributions: torch.Tensor,
    image: torch.Tensor,
    cmap: str,
    center_at_zero: bool,
    title: Optional[str],
    overlay: bool,
):
    vmax = (
        attributions.abs().max().item()
        if center_at_zero
        else attributions.max().item()
    )
    if overlay:
        ax.imshow(image, alpha=0.5)
    vmin = -vmax if center_at_zero else attributions.min().item()
    alpha = 0.5 if overlay else 1.0
    img = ax.imshow(attributions, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    fig.colorbar(img, ax=ax)
    if title is not None:
        ax.set_title(title)


def visualize_attributions(
    attributions: Dict[str, torch.Tensor],
    image: torch.Tensor,
    cmap: str = "bwr",
    center_at_zero: bool = True,
    overlay = False,
) -> Figure:
    """Visualize attributions.
    Attributions can be visualized by overlaying them on the original
    image, by plotting them as a heatmap, or by plotting the original image
    with a transparency mask over it, making pixels with higher attribution
    values more visible.

    The shape of images and attributions is assumed to be (N, C, H, W),
    where N is the number of samples, C is the number of channels, and
    H and W are the height and width of the images. The channel dimension is
    eliminated by averaging over it.

    Parameters
    ----------
    attributions : Dict[str, torch.Tensor]
        Dictionary mapping method names to attributions. The attributions
        should have shape (C, H, W).
    image : torch.Tensor, optional
        Original image. Shape: (C, H, W), by default None.
    cmap : str, optional
        Colormap to use for plotting the heatmap, by default "bwr".
    center_at_zero : bool, optional
        Whether to center the colormap at zero, making a zero attribution
        value correspond to white in a diverging colormap. By default True.
    overlay : bool, optional
        Whether to overlay the attributions on the original image, by default
        False.
    """
    # Checking inputs
    num_methods = len(attributions.keys())
    
    n_rows = num_methods // 4 + 1
    n_cols = 4

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    if num_methods != 1:
        axs = axs.flatten()
    
    # Plot original image
    axs[0].imshow(image)
    axs[0].set_title("Original image")

    # Plot heatmaps
    for idx, method_name in enumerate(attributions.keys()):
        if num_methods != 1:
            ax = axs[idx + 1]
        else:
            ax = axs
        assert isinstance(ax, plt.Axes)
        _plot_heatmap(
            fig,
            ax,
            attributions[method_name].mean(dim=0),
            image,
            cmap=cmap,
            center_at_zero=center_at_zero,
            title=method_name,
            overlay=overlay
        )
    
    if n_rows * n_cols > num_methods + 1:
        for ax in axs[num_methods + 1:]:
            ax.remove()

    return fig
