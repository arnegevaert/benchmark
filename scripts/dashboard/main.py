import argparse
import webbrowser
from attrbench.suite import Result
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import numpy as np
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    args = parser.parse_args()

    hdf_obj = Result.load_hdf(args.file)
    app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

    app_children = [
        html.H1(args.file)
    ]

    for metric_name in hdf_obj.get_metrics():
        metric_data = hdf_obj.data[metric_name]
        method_names = list(metric_data.keys())
        # If the metric has a "col_index" property, use a line plot
        if "col_index" in hdf_obj.metadata[metric_name].keys():
            index = hdf_obj.metadata[metric_name]["col_index"]
            plot_df = pd.DataFrame(columns=method_names, index=index)
            for method_name in method_names:
                plot_df[method_name] = np.average(metric_data[method_name], axis=0)
            fig = px.line(plot_df)
            app_children.append(html.H2(metric_name))
            app_children.append(dcc.Graph(
                id=metric_name,
                figure=fig
            ))
        # Otherwise, use a box plot
        else:
            plot_df = pd.concat(metric_data, axis=1)
            plot_df.columns = plot_df.columns.get_level_values(0)
            fig = px.box(plot_df)
            app_children.append(html.H2(metric_name))
            app_children.append(dcc.Graph(
                id=metric_name,
                figure=fig
            ))

    app.layout = html.Div(children=app_children)

    port = 9000
    app.run_server(port=port)
    webbrowser.open_new(f"localhost:{port}")