import dash_table
import dash_html_components as html
from dash_table.Format import Format
from attrbench.suite.dashboard.components import Component


class PValueTable(Component):
    def __init__(self, pvalues, id):
        self.pvalues = pvalues
        self.id = id

    def render(self) -> html.Div:
        return html.Div(dash_table.DataTable(
            id=self.id,
            columns=[{"name": "Method", "id": f"method", "type": "text"},
                     {"name": "p-value", "id": f"pvalue", "type": "numeric", "format": Format(precision=3)}],
            data=self.pvalues,
            style_data_conditional=[
                {"if": {"filter_query": "{pvalue} < 0.05"}, "backgroundColor": "lightgreen"},
                {"if": {"filter_query": "{pvalue} >= 0.05"}, "backgroundColor": "lightred"},
            ],
            style_table={"width": "30%"}
        ))

