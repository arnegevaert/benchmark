import dash_html_components as html

from attrbench.suite.dashboard.components.pages import Page
from attrbench.suite.dashboard.components.plots import PValueTable
from scipy.stats import shapiro


class NormalityPage(Page):
    def __init__(self, result_obj):
        super().__init__(result_obj)

    def render(self) -> html.Div:
        result = [html.P("Normality of results is tested using Shapiro-Wilk test. "
                         "Significance (p < 0.05) means that the data is NOT normally distributed.")]
        for metric_name in self.result_obj.get_metrics():
            metric_shape = self.result_obj.metadata[metric_name]["shape"]
            if metric_shape[0] > 1:
                result.append(html.H2(metric_name))
                pvalues = []
                for method_name in self.result_obj.get_methods():
                    method_data = self.result_obj.data[metric_name][method_name]
                    method_pvalues = {"method": method_name}
                    for col in range(metric_shape[1]):
                        col_data = method_data[col]
                        stat, pvalue = shapiro(col_data)
                        method_pvalues[f"p-value {col}"] = pvalue
                    pvalues.append(method_pvalues)
                result.append(PValueTable(pvalues, id=f"table-normality-{metric_name}").render())
        return html.Div(result)
