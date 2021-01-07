import dash_html_components as html


class Component:
    def render(self) -> html.Div:
        raise NotImplementedError
