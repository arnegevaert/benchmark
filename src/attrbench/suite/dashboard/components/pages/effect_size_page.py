import dash_html_components as html
import pandas as pd
from scipy.stats import wilcoxon

from attrbench.suite.dashboard.components.pages import Page
from attrbench.suite.dashboard.components.plots import EffectSizePlot, PValueTable


class EffectSizePage(Page):
    def __init__(self, result_obj):
        super().__init__(result_obj)

    def render(self) -> html.Div:
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
                result.append(EffectSizePlot(df, "Random", id=f"{metric_name}-effect-size").render())
                pvalues = []
                for method_name in self.result_obj.get_methods():
                    if method_name != "Random":
                        statistic, pvalue = wilcoxon(df[method_name].to_numpy(), df["Random"].to_numpy(), alternative="two-sided")
                        pvalues.append({"method": method_name, "p-value": pvalue})
                result.append(PValueTable(pvalues, id=f"table-pvalues-{metric_name}").render())
        return html.Div(result)
