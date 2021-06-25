import base64
from io import BytesIO
import dash_html_components as html
import attrbench.suite.dashboard.util as util
from attrbench.suite.plot.wilcoxon_summary_plot import WilcoxonSummaryPlot
from attrbench.lib.stat import wilcoxon_tests
import pandas as pd
from plotly import express as px
import plotly.graph_objects as go

def wilcoxon_summary_plot(df):
    fig =WilcoxonSummaryPlot(df).render()
    plot_img = BytesIO()
    fig.savefig(plot_img, format="png")
    plot_img.seek(0)
    encoded = base64.b64encode(plot_img.read()).decode("ascii").replace("\n", "")
    return html.Div([html.Img(src=f"data:image/png;base64,{encoded}")])

def plotly_wilcoxon_summary_plot(dfs):
    effect_sizes, p_vals ={},{}
    ALPHA = 0.01
    for metric_name, (df, inverted) in dfs.items():
        # Compute effect sizes and p-values
        mes, mpv = wilcoxon_tests(df, inverted)
        effect_sizes[metric_name]=mes
        p_vals[metric_name]=mpv
    p_vals = pd.DataFrame(p_vals)
    effect_sizes = pd.DataFrame(effect_sizes).abs()
    norm_effect_sizes = (effect_sizes - effect_sizes.min()) / (effect_sizes.max() - effect_sizes.min())
    effect_sizes[p_vals > 0.01] = None
    effect_sizes=effect_sizes.transpose()
    # effect_sizes=effect_sizes.reset_index()
    # effect_sizes = pd.melt(effect_sizes, id_vars='index')
    # effect_sizes.columns = ["method", "metric", "value"]

    # fig = px.imshow(effect_sizes)

    fig = go.Figure(data=go.Heatmap(z=effect_sizes,x=effect_sizes.columns,y=effect_sizes.index))

    return fig

