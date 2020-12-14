import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from attrbench.suite.dashboard.sidebar import Sidebar
from attrbench.suite.dashboard.pages import OverviewPage, CorrelationsPage, ClusteringPage, SamplesAttributionsPage


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

        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        self.pages = {
            "/overview": OverviewPage(self.result_obj),
            "/correlations": CorrelationsPage(self.result_obj),
            "/clustering": ClusteringPage(),
            "/samples_attributions": SamplesAttributionsPage()
        }
        self.root = "/overview"
        self.sidebar = Sidebar(self.app, path_titles={
            "/overview": "Overview",
            "/correlations": "Correlations",
            "/clustering": "Clustering",
            "/samples_attributions": "Samples/Attributions"
        })
        self.page_content = html.Div(id="page-content", style=Dashboard._CONTENT_STYLE)

    def render_page_content(self, pathname):
        content = [html.H1(self.title)]
        if pathname == "/":
            rendered = self.pages[self.root].render()
            content.extend(rendered if type(rendered) == list else [rendered])
        elif pathname in self.pages:
            rendered = self.pages[pathname].render()
            content.extend(rendered if type(rendered) == list else [rendered])
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

    def run(self):
        self.app.layout = html.Div(
            [dcc.Location(id="url")] + self.sidebar.render() + [self.page_content]
        )

        self.app.callback(
            Output("page-content", "children"),
            [Input("url", "pathname")])(self.render_page_content)

        self.app.run_server(port=self.port)

