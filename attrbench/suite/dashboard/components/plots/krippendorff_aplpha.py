import base64
from io import BytesIO
import dash_html_components as html
import attrbench.suite.dashboard.util as util
from attrbench.suite.plot.krippendorff_alpha import KrippendorffAlphaPlot
import pandas as pd
from plotly import express as px

def krippendorff_aplpha_plot(df):
    fig =KrippendorffAlphaPlot(df).render()
    plot_img = BytesIO()
    fig.savefig(plot_img, format="png")
    plot_img.seek(0)
    encoded = base64.b64encode(plot_img.read()).decode("ascii").replace("\n", "")
    return html.Div([html.Img(src=f"data:image/png;base64,{encoded}")])

def plotly_krippendroff_alpha_plot(dfs):
    ka = {}
    for metric_name, (df, inverted)  in dfs.items():
        ka[metric_name] = util.krippendorff_alpha(df.to_numpy())
    ka = pd.DataFrame(ka, index=["Krippendorff Alpha"]).transpose()
    return px.bar(ka, orientation='h')