import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from attrbench.suite.dashboard.component import Sidebar
from attrbench.suite.dashboard.pages import OverviewPage


class Dashboard:
    # the styles for the main content position it to the right of the sidebar and
    # add some padding.
    _CONTENT_STYLE = {
        "margin-left": "22rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }

    def __init__(self, result_obj, title, port=9000):
        self.result_obj = result_obj
        self.title = title
        self.port = port
        self.sidebar = Sidebar()
        self.content = html.Div(id="page-content", style=Dashboard._CONTENT_STYLE)
        self.overview_page = OverviewPage(self.result_obj)

    def run(self):
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.layout = html.Div(
            [dcc.Location(id="url")] + self.sidebar.render() + [self.content]
        )

        """
        app_children = 

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
            content = [html.H1(self.title)]
            if pathname in ["/", "/overview"]:
                content += self.overview_page.render()
            elif pathname == "/correlations":
                content.append(html.P("Correlations page"))
            elif pathname == "/clustering":
                content.append(html.P("Clustering page"))
            else:
                # If the user tries to reach a different page, return a 404 message
                return dbc.Jumbotron(
                    [
                        html.H1("404: Not found", className="text-danger"),
                        html.Hr(),
                        html.P(f"The pathname {pathname} was not recognised..."),
                    ]
                )
            return content
        app.run_server(port=self.port)

