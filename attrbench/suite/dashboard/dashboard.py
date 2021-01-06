import dash
from attrbench.suite.dashboard.sidebar import Sidebar
from attrbench.suite.dashboard.pages import *


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
        self.app.config["suppress_callback_exceptions"] = True
        self.pages = {
            "/overview": (OverviewPage(self.result_obj), "Overview"),
            "/correlations": (CorrelationsPage(self.result_obj), "Correlations"),
            "/clustering": (ClusteringPage(self.result_obj), "Clustering"),
            "/samples_attributions": (SamplesAttributionsPage(self.result_obj, self.app), "Samples/Attributions"),
            "/detail": (DetailPage(self.result_obj, self.app), "Detail"),
            "/effect_size": (EffectSizePage(self.result_obj), "Effect Size")
        }

        self.root = "/overview"
        self.sidebar = Sidebar(self.app, path_titles={path: self.pages[path][1] for path in self.pages.keys()})
        self.page_content = html.Div(id="page-content", style=Dashboard._CONTENT_STYLE)

        # Callback to handle routing
        self.app.callback(
            Output("page-content", "children"),
            [Input("url", "pathname")])(self.render_page_content)

    def render_page_content(self, pathname):
        content = [html.H1(self.title)]
        if pathname == "/":
            rendered = self.pages[self.root][0].render()
            content.extend(rendered if type(rendered) == list else [rendered])
        elif pathname in self.pages:
            rendered = self.pages[pathname][0].render()
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

        self.app.run_server(port=self.port)

