from bokeh import plotting, palettes, models, layouts
import itertools
import torch
import numpy as np


class Report:
    """
    Represents and renders a report consisting of a single summary plot (e.g. line plot with results)
    followed by a set of example plots for each considered method (e.g. perturbed images)
    """
    def __init__(self, methods):
        self.methods = methods
        self.method_examples = {method: [] for method in methods}
        self.summary_plot = plotting.figure()
        self.colors = itertools.cycle(palettes.Dark2_8)

    def add_method_example_row(self, method, examples):
        self.method_examples[method].append(examples)

    def reset_method_examples(self, method):
        self.method_examples[method] = []

    def add_summary_line(self, x, y, label):
        self.summary_plot.line(x, y, legend_label=label, color=next(self.colors), line_width=3)

    def render(self):
        plots = [[self.summary_plot]]
        for method in self.methods:
            plots.append([models.Div(text=f"<h1>{method}</h1>", sizing_mode="stretch_width")])
            for row in self.method_examples[method]:
                plots.append(self._plot_images_row(np.stack(row, axis=0)))
        plotting.show(layouts.layout(plots))

    def save(self, location):
        pass

    def load(self, location):
        pass

    @staticmethod
    def _plot_images_row(imgs):
        if type(imgs) == torch.Tensor:
            imgs = imgs.detach().numpy()
        is_grayscale = imgs.shape[1] == 1
        n_imgs = imgs.shape[0]
        concat_imgs = []
        for i in range(n_imgs):
            img = imgs[i]
            # Convert range to 0..255 uint32
            img = (img - np.min(img))/(np.max(img) - np.min(img))
            img *= 255
            img = np.array(img, dtype=np.uint32)

            # If grayscale, just provide rows/cols. If RGBA, convert to RGBA
            if is_grayscale:
                img = np.squeeze(img)
            else:
                img = img.transpose((1, 2, 0))
                rgba_img = np.empty(shape=(img.shape[0], img.shape[1], 4), dtype=np.uint8)
                rgba_img[:, :, :3] = img
                rgba_img[:, :, 3] = 255
                img = rgba_img
            img = np.flip(img, axis=0)
            concat_imgs.append(img)
        concat_imgs = np.concatenate(concat_imgs, axis=-1)
        p = plotting.figure(width=200*n_imgs, height=200)
        p.toolbar_location = None
        p.axis.visible = False
        p.grid.visible = False
        plot_fn = p.image if is_grayscale else p.image_rgba
        plot_fn([concat_imgs], x=0, y=0, dw=1, dh=1)
        return p
