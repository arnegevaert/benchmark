import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


class Component:


    def render(self):
        raise NotImplementedError


class Sidebar(Component):
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

    def render(self):
        return [html.Div(
            [
                html.H2("Dashboard", className="display-4"),
                html.Hr(),
                dbc.Nav(
                    [
                        dbc.NavLink("Overview", href="/overview", id="overview-link"),
                        dbc.NavLink("Correlations", href="/correlations", id="correlations-link"),
                        dbc.NavLink("Clustering", href="/clustering", id="clustering-link"),
                    ],
                    vertical=True,
                    pills=True,
                ),
            ],
            style=Sidebar._STYLE,
        )]
