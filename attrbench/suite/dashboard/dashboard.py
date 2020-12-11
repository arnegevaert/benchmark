import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from attrbench.suite.dashboard.components import *
from dash.dependencies import Input, Output


class Dashboard:
    # the style arguments for the sidebar. We use position:fixed and a fixed width
    _SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    }

    # the styles for the main content position it to the right of the sidebar and
    # add some padding.
    _CONTENT_STYLE = {
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }

    def __init__(self, result_obj, title, port=9000):
        self.result_obj = result_obj
        self.title = title
        self.port = port
        self.sidebar = self._build_sidebar()
        self.content = html.Div(id="page-content", style=Dashboard._CONTENT_STYLE)

    def _build_sidebar(self):
        return html.Div(
            [
                html.H2("Sidebar", className="display-4"),
                html.Hr(),
                html.P(
                    "A simple sidebar layout with navigation links", className="lead"
                ),
                dbc.Nav(
                    [
                        dbc.NavLink("Page 1", href="/page-1", id="page-1-link"),
                        dbc.NavLink("Page 2", href="/page-2", id="page-2-link"),
                        dbc.NavLink("Page 3", href="/page-3", id="page-3-link"),
                    ],
                    vertical=True,
                    pills=True,
                ),
            ],
            style=Dashboard._SIDEBAR_STYLE,
        )

    def run(self):
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.layout = html.Div([dcc.Location(id="url"), self.sidebar, self.content])

        """
        app_children = [
            html.H1(self.title)
        ]

        for metric_name in self.result_obj.get_metrics():
            app_children.append(html.H2(metric_name))
            if "col_index" in self.result_obj.metadata[metric_name].keys():
                plot = Lineplot(self.result_obj.data[metric_name],
                                x_ticks=self.result_obj.metadata[metric_name]["col_index"])
            else:
                plot = Boxplot(self.result_obj.data[metric_name])
            app_children.append(dcc.Graph(
                id=metric_name,
                figure=plot.render()
            ))

        app.layout = html.Div(children=app_children)
        """

        @app.callback(
            [Output(f"page-{i}-link", "active") for i in range(1, 4)],
            [Input("url", "pathname")]
        )
        def toggle_active_links(pathname):
            if pathname == "/":
                return True, False, False
            return [pathname == f"/page-{i}" for i in range(1, 4)]

        @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
        def render_page_content(pathname):
            if pathname in ["/", "/page-1"]:
                return html.P("This is the content of page 1!")
            elif pathname == "/page-2":
                return html.P("This is the content of page 2. Yay!")
            elif pathname == "/page-3":
                return html.P("Oh cool, this is page 3!")
            # If the user tries to reach a different page, return a 404 message
            return dbc.Jumbotron(
                [
                    html.H1("404: Not found", className="text-danger"),
                    html.Hr(),
                    html.P(f"The pathname {pathname} was not recognised..."),
                ]
            )
        app.run_server(port=self.port)

