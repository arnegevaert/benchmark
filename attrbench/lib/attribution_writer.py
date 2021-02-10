import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib import colors
import matplotlib.pyplot as plt


class AttributionWriter(SummaryWriter):
    def __init__(self, log_dir=None, comment='', mean=None, std=None, purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix=''):
        super().__init__(log_dir=log_dir, comment=comment, purge_step=purge_step, max_queue=max_queue,
                         flush_secs=flush_secs, filename_suffix=filename_suffix)

        self.mean = mean
        self.std = std

        self.batch_nr=0
        self.method_name=None

    def set_method_name(self, name):
        self.method_name = name
    def increment_batch(self):
        self.batch_nr +=1

    def _scale_images(self, img_tensor):
        return (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

    def _clip_image(self, img_tensor):
        res = img_tensor
        res[res < 0.] = 0.
        res[res > 1.] = 1.
        return res

    def _normalize_images(self, img_tensor):
        dtype = img_tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=img_tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=img_tensor.device)
        img_tensor = (img_tensor * std.view(1, -1, 1, 1)) + mean.view(1, -1, 1, 1)
        return img_tensor

    def _attrshow(self, attrs):
        npattrs = attrs.squeeze().detach().cpu().numpy()  # [batch_size, rows, cols]
        npattrs = np.concatenate(npattrs, axis=-1)
        min_value = min(np.min(npattrs), -.01)
        max_value = max(np.max(npattrs), .01)
        divnorm = colors.TwoSlopeNorm(vmin=min_value, vcenter=0., vmax=max_value)

        fig, ax = plt.subplots()

        cs = ax.imshow(npattrs, cmap="bwr", norm=divnorm)
        fig.colorbar(cs, orientation="horizontal")
        plt.tight_layout()
        return fig

    def add_images(self, tag, img_tensor, **kwargs):
        tag = tag + "/{}/{}".format(self.method_name, self.batch_nr)
        if 'samples' in tag:
            # normalize images
            if self.mean and self.std:
                img_tensor = self._normalize_images(img_tensor)
            # scale values to [0,1] for plotting if no std or mean were given
            else:
                img_tensor = self._scale_images(img_tensor)

        elif 'attr' in tag:
            if img_tensor.shape[-3] > 1:
                img_tensor = self._scale_images(img_tensor)
            # if attributions have one channel, use the figure method
            else:
                fig = self._attrshow(img_tensor)
                super().add_figure(tag, fig)
                return

        if 'perturb' in tag:
            img_tensor = self._scale_images(img_tensor)

        super().add_images(tag, img_tensor, **kwargs)
