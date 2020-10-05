import argparse
import numpy as np
from experiments.benchmark import load_results, get_metric
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column
from bokeh.palettes import Category10_10 as palette
from bokeh.models import Legend, Div


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str)
    args = parser.parse_args()

    all_metrics = ["insertion", "deletion", "infidelity", "max-sens", "s-impact", "impact", "sens-n"]

    result_data = load_results(args.dir)

    for method in result_data:
        print(method)
        for res in result_data[method]:
            print(f"    {res}: {result_data[method][res].shape}")

    output_file("report.html")
    figures = []
    for metric in all_metrics:
        legend_data = []
        data = get_metric(result_data, metric)
        p = figure(title=metric, x_axis_label="x", y_axis_label="y", plot_width=1000, plot_height=500)
        col_iter = iter(palette)
        for method in data:
            col = next(col_iter)
            m_data = data[method]
            if len(m_data.shape) > 1:
                mean = np.mean(m_data, axis=0)
                sd = np.std(m_data, axis=0)
                l = p.line(x=np.arange(m_data.shape[1]), y=mean, color=col, line_width=2)
                a = p.varea(x=np.arange(m_data.shape[1]),
                            y1=(mean - (1.96 * sd / np.sqrt(m_data.shape[0]))),
                            y2=(mean + (1.96 * sd / np.sqrt(m_data.shape[0]))),
                            color=col, alpha=0.2)
                legend_data.append((method, [l, a]))
            else:
                l = p.line(x=np.arange(m_data.shape[0]), y=m_data, color=col, line_width=2)
                legend_data.append((method, [l]))
        legend = Legend(items=legend_data)
        legend.click_policy = "hide"
        p.add_layout(legend, "right")
        figures.append(p)
    show(column([Div(text=f"<h1>Directory: {args.dir}</h1>")] + figures))
