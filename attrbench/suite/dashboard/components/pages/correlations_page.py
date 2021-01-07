import dash_html_components as html
import numpy as np

from attrbench.suite.dashboard.components.pages import Page
from attrbench.suite.dashboard.components.plots import InterMetricCorrelationPlot, InterMethodCorrelationPlot, BarPlot
from attrbench.suite.dashboard.util import krippendorff_alpha


class CorrelationsPage(Page):
    def __init__(self, result_obj):
        super().__init__(result_obj)
        self.rendered_contents = None

    def render(self) -> html.Div:
        if not self.rendered_contents:
            result = []
            # Krippendorff Alpha
            result.append(html.H2("Krippendorff Alpha"))
            names, values = [], []
            metrics = [m for m in self.result_obj.get_metrics() if self.result_obj.metadata[m]["shape"][0] > 1]
            for metric_name in metrics:
                metric_data = self.result_obj.data[metric_name]
                metric_metadata = self.result_obj.metadata[metric_name]
                names.append(metric_name)
                data = np.stack(
                    [metric_data[method_name].mean(axis=1).to_numpy()
                     for method_name in self.result_obj.get_methods()],
                    axis=1)
                values.append(krippendorff_alpha(np.argsort(data)))
            result.append(BarPlot(values, names, id="krippendorff-alpha").render())

            # Inter-metric correlation
            result.append(html.H2("Inter-method correlations"))
            for method_name in self.result_obj.get_methods():
                result.append(html.H3(method_name))
                plot = InterMethodCorrelationPlot(self.result_obj, method_name, id=f"{method_name}-metric-corr")
                result.append(plot.render())

            # Inter-method correlation
            result.append(html.H2("Inter-metric correlations"))
            metrics = [m for m in self.result_obj.get_metrics()
                       if self.result_obj.metadata[m]["shape"][0] == self.result_obj.num_samples]
            for metric_name in metrics:
                result.append(html.H3(metric_name))
                plot = InterMetricCorrelationPlot(self.result_obj, metric_name, id=f"{metric_name}-method-corr")
                result.append(plot.render())
            self.rendered_contents = html.Div(result)
        return self.rendered_contents
