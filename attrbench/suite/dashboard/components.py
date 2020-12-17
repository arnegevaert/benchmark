import numpy as np
from plotly import express as px
import dash_bootstrap_components as dbc
import dash_core_components as dcc


class Component:
    def render(self):
        raise NotImplementedError


class SampleAttributionsComponent(Component):
    def __init__(self, image, attrs, id, method_names):
        self.image = np.squeeze(np.moveaxis(image, 0, 2))
        self.attrs = attrs
        self.id = id
        self.method_names = method_names
        self.color_image = len(self.image.shape) > 2

    def render(self):
        image_fig = px.imshow(self.image, color_continuous_scale="gray" if not self.color_image else None,
                              width=300, height=300)
        image_fig.update_xaxes(showticklabels=False)
        image_fig.update_yaxes(showticklabels=False)
        image_fig.update_layout(margin=dict(l=0, r=0, t=5, b=5))

        attrs_fig = px.imshow(self.attrs, color_continuous_scale="gray", facet_col=0,
                              height=300, width=self.attrs.shape[0]*300, labels={"facet_col": "method"})
        for j, method_name in enumerate(self.method_names):
            attrs_fig.layout.annotations[j]["text"] = method_name
        attrs_fig.update_xaxes(showticklabels=False)
        attrs_fig.update_yaxes(showticklabels=False)
        attrs_fig.update_layout(margin=dict(l=0, r=0, t=15, b=0))

        return dbc.Row([
            dbc.Col(dcc.Graph(
                id=f"orig-image-{self.id}",
                figure=image_fig
            ), className="col-md-3"),
            dbc.Col(dcc.Graph(
                id=f"attrs-{self.id}",
                figure=attrs_fig
            ), className="col-md-6"),
        ])
