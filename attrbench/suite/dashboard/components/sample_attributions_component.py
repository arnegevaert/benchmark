import dash_bootstrap_components as dbc
import dash_core_components as dcc
import numpy as np
from plotly import express as px

from attrbench.suite.dashboard.components import Component


class SampleAttributionsComponent(Component):
    def __init__(self, image, attrs, id, method_names):
        self.image = image
        self.attrs = attrs
        self.id = id
        self.method_names = method_names

    def render(self):
        cols = []
        plot_image = np.squeeze(np.moveaxis(self.image, 0, 2))
        color_image = self.image.shape[0] == 3
        print(plot_image.shape)
        if color_image:
            plot_image = (plot_image - np.min(plot_image)) / (np.max(plot_image) - np.min(plot_image)) * 255
            plot_image = plot_image.astype(np.uint8)
        image_fig = px.imshow(plot_image, color_continuous_scale="gray" if not color_image else None,
                              width=300, height=300)
        image_fig.update_xaxes(showticklabels=False)
        image_fig.update_yaxes(showticklabels=False)
        image_fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        cols.append(dbc.Col(dcc.Graph(
            id=f"orig-image-{self.id}",
            figure=image_fig
        ), className="col-md-3"))

        for i in range(self.attrs.shape[0]):
            if self.attrs.shape[1] == 1:
                # Per-pixel attributions, grayscale
                attrs_fig = px.imshow(np.squeeze(self.attrs[i, ...]), color_continuous_scale="gray",
                                      height=300, width=300, title=self.method_names[i])
            elif self.attrs.shape[1] == 3:
                # Per-channel attributions, color
                plot_attrs = np.moveaxis(self.attrs[i, ...], 0, 2)
                plot_attrs = (plot_attrs - np.min(plot_attrs)) / (np.max(plot_attrs) - np.min(plot_attrs)) * 255
                plot_attrs = plot_attrs.astype(np.uint8)
                attrs_fig = px.imshow(plot_attrs, height=300, width=300, title=self.method_names[i])
            else:
                raise ValueError(f"Invalid attributions shape: {self.attrs.shape}")
            attrs_fig.update_xaxes(showticklabels=False)
            attrs_fig.update_yaxes(showticklabels=False)
            attrs_fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            cols.append(dbc.Col(dcc.Graph(
                id=f"attrs-{self.id}-{self.method_names[i]}",
                figure=attrs_fig
            ), className="col-md-3"))
        return dbc.Row(cols, className="mt-1")
