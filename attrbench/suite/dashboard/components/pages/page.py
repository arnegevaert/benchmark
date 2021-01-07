from attrbench.suite.dashboard.components import Component
import dash_html_components as html


class Page(Component):
    def __init__(self, result_obj):
        self.result_obj = result_obj

    def render(self) -> html.Div:
        raise NotImplementedError()
