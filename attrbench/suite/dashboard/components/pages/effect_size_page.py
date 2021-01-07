import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
from dash_table.Format import Format
from scipy.stats import ttest_rel

from attrbench.suite.dashboard.components.pages import Page
from attrbench.suite.dashboard.components.plots import EffectSizePlot


class EffectSizePage(Page):
    def __init__(self, result_obj):
        super().__init__(result_obj)

    def render(self):
        result = []
        for metric_name in self.result_obj.get_metrics():
            metric_shape = self.result_obj.metadata[metric_name]["shape"]
            if metric_shape[0] == self.result_obj.num_samples:
                result.append(html.H2(metric_name))
                data = {}
                for method_name in self.result_obj.get_methods():
                    method_data = self.result_obj.data[metric_name][method_name]
                    data[method_name] = method_data.mean(axis=1)
                df = pd.concat(data, axis=1)
                plot = EffectSizePlot(df, "Random")  # TODO choose baseline name
                result.append(dcc.Graph(
                    id=f"{metric_name}-effect-size",
                    figure=plot.render()
                ))
                pvalues = []
                for method_name in self.result_obj.get_methods():
                    if method_name != "Random":
                        statistic, pvalue = ttest_rel(df[method_name].to_numpy(), df["Random"].to_numpy())
                        pvalues.append({"method": method_name, "pvalue": pvalue})
                result.append(dash_table.DataTable(
                    id=f"table-pvalues-{metric_name}",
                    columns=[{"name": "Method", "id": f"method", "type": "text"},
                             {"name": "p-value", "id": f"pvalue", "type": "numeric", "format": Format(precision=3)}],
                    data=pvalues,
                    style_data_conditional=[
                        {"if": {"filter_query": "{pvalue} < 0.05"}, "backgroundColor": "lightgreen"},
                        {"if": {"filter_query": "{pvalue} >= 0.05"}, "backgroundColor": "lightred"},
                    ],
                    style_table={"width": "30%"}
                ))
        return result
