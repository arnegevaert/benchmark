import dash_table
import dash_html_components as html
from dash_table.Format import Format
from attrbench.suite.dashboard.components import Component


class PValueTable(Component):
    def __init__(self, pvalues, id):
        self.pvalues = pvalues
        self.id = id

    def render(self) -> html.Div:
        columns = [{"name": "Method", "id": f"method", "type": "text"}]
        style_cond = []
        for col in self.pvalues[0].keys():
            if col != "method":
                columns.append({"name": col, "id": col, "type": "numeric", "format": Format(precision=3)})
                style_cond.append({"if": {"column_id": col, "filter_query": "{" + col + "} < 0.05"}, "backgroundColor": "lightgreen"})
                style_cond.append({"if": {"column_id": col, "filter_query": "{" + col + "} >= 0.05"}, "backgroundColor": "pink"},)

        return html.Div(dash_table.DataTable(
            id=self.id,
            columns=columns,
            data=self.pvalues,
            style_data_conditional=style_cond,
            style_table={"width": "30%"}
        ))
