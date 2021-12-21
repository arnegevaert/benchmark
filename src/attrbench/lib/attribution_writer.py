import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib import colors
import matplotlib.pyplot as plt


def _scale_images(img_tensor):
    return torch.true_divide((img_tensor - img_tensor.min()), (img_tensor.max() - img_tensor.min()))


def _clip_image(img_tensor):
    res = img_tensor
    res[res < 0.] = 0.
    res[res > 1.] = 1.
    return res


def _attrshow(attrs):
    npattrs = attrs.squeeze()  # [batch_size, rows, cols]
    if len(npattrs.shape) == 2:
        # If the batch had only 1 sample, we need to add back the original batch dim
        npattrs = npattrs[np.newaxis, ...]
    npattrs = np.concatenate(npattrs, axis=-1)
    min_value = min(np.min(npattrs), -.01)
    max_value = max(np.max(npattrs), .01)
    divnorm = colors.TwoSlopeNorm(vmin=min_value, vcenter=0., vmax=max_value)

    fig, ax = plt.subplots()

    cs = ax.imshow(npattrs, cmap="bwr", norm=divnorm)
    fig.colorbar(cs, orientation="horizontal")
    plt.tight_layout()
    return fig


class AttributionWriter(SummaryWriter):
    def __init__(self, log_dir: str, method_name: str = None, comment='', mean=None, std=None, purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix=''):
        super().__init__(log_dir=log_dir, comment=comment, purge_step=purge_step, max_queue=max_queue,
                         flush_secs=flush_secs, filename_suffix=filename_suffix)

        self.mean = mean
        self.std = std

        self.batch_nr = 0
        self.method_name = method_name

    def increment_batch(self):
        self.batch_nr += 1

    def _normalize_images(self, img_tensor):
        dtype = img_tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=img_tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=img_tensor.device)
        img_tensor = (img_tensor * std.view(1, -1, 1, 1)) + mean.view(1, -1, 1, 1)
        return img_tensor

    def add_images(self, tag, img_tensor, global_step=None, **kwargs):
        if self.mean and self.std:
            img_tensor = self._normalize_images(img_tensor)
        # scale values to [0,1] for plotting if no std or mean were given
        else:
            img_tensor = _scale_images(img_tensor)
        super().add_images(tag, img_tensor, global_step=global_step, **kwargs)

    def add_attribution(self, tag, img_tensor, global_step=None):
        if self.method_name is not None:
            tag = f"{tag}/{self.method_name}"
        # If attributions have 1 channel, use the image method
        if img_tensor.shape[-3] > 1:
            img_tensor = _scale_images(img_tensor)
            super().add_images(tag, img_tensor, global_step=global_step)
        # if attributions have one channel, use the figure method
        else:
            fig = _attrshow(img_tensor)
            super().add_figure(tag, fig, global_step=global_step)
