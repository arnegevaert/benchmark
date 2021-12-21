import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from attrbench.suite.dashboard.components import Component


class SidebarComponent(Component):
    # the style arguments for the sidebar. We use position:fixed and a fixed width
    _STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "20rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    }

    def __init__(self, app, path_titles):
        super().__init__()
        self.app = app
        self.path_titles = path_titles
        self.path_ids = {
            key: f"{key[1:]}-link" for key in path_titles
        }

        # Callback for active tab
        app.callback(
            [Output(self.path_ids[key], "active") for key in self.path_ids],
            [Input("url", "pathname")])(self.toggle_active_links)

    def toggle_active_links(self, pathname):
        if pathname == "/":
            return tuple([True] + [False] * (len(self.path_titles) - 1))
        return [pathname == n for n in self.path_titles.keys()]

    def render(self):
        navlinks = [
            dbc.NavLink(self.path_titles[key], href=key, id=self.path_ids[key]) for key in self.path_titles
        ]
        return html.Div(
            [
                html.H2("Dashboard", className="display-4"),
                html.Hr(),
                dbc.Nav(navlinks, vertical=True, pills=True),
            ],
            style=SidebarComponent._STYLE,
        )
