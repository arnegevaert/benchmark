import base64
from io import BytesIO
import dash_html_components as html
from attrbench.suite.plot.cluster_plot import ClusterPlot


def cluster_map(dfs) -> html.Div:
    fig =ClusterPlot(dfs).render()
    plot_img = BytesIO()
    fig.savefig(plot_img, format="png")
    plot_img.seek(0)
    encoded = base64.b64encode(plot_img.read()).decode("ascii").replace("\n", "")
    return html.Div([html.Img(src=f"data:image/png;base64,{encoded}")])
