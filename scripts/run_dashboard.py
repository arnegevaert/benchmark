import argparse
from attrbench.suite import Result
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    args = parser.parse_args()

    hdf_obj = Result.load_hdf(args.file)

    app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

    df = hdf_obj.data["deletion"]["Gradient"]
    fig = px.line(df.transpose())

    app.layout = html.Div(children=[
        html.H1("Hello dash"),
        html.Div(children="""
            Dash example
        """),
        dcc.Graph(
            id="example-graph",
            figure=fig
        )
    ])
    app.run_server()
