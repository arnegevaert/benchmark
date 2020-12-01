import argparse
import numpy as np
import webbrowser
import base64
from io import BytesIO
import os
import matplotlib.pyplot as plt
from experiments.independent import Result, correlation_heatmap
from scipy.stats import spearmanr
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column
from bokeh.palettes import Category10_10 as palette
from bokeh.models import Legend, Div
import holoviews as hv
from scipy.cluster import hierarchy
from sklearn import preprocessing
hv.extension("bokeh")


def _interval_metric(a, b):
    return (a - b) ** 2


def krippendorff_alpha(data):
    # Assumptions: no missing values, interval metric, data is numpy array ([observers, samples])
    # Assuming no missing values, each column is a unit, and the number of pairable values is m*n
    pairable_values = data.shape[0] * data.shape[1]

    # Calculate observed disagreement
    observed_disagreement = 0.
    for col in range(data.shape[1]):
        unit = data[:, col].reshape(1, -1)
        observed_disagreement += np.sum(_interval_metric(unit, unit.T))
    observed_disagreement /= (pairable_values * (data.shape[0] - 1))

    # Calculate expected disagreement
    expected_disagreement = 0.
    for col1 in range(data.shape[1]):
        unit1 = data[:, col1].reshape(1, -1)
        for col2 in range(data.shape[1]):
            unit2 = data[:, col2].reshape(1, -1)
            expected_disagreement += np.sum(_interval_metric(unit1, unit2.T))
    expected_disagreement /= (pairable_values * (pairable_values - 1))
    return 1. - (observed_disagreement / expected_disagreement)


def convert_to_html(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    html_str = f"<div><img src=data:image/png;base64,{data}></div>"
    return html_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    parser.add_argument("--bs-size", type=int, default=0)
    parser.add_argument("--bs-amt", type=int, default=0)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--part", type=str, default="all", choices=["all", "report", "consistency", "cluster",
                                                                    "global_cluster", "boxplots"])
    args = parser.parse_args()
    result_obj = Result(args.dir)

    """
    PART 1: GENERAL REPORT
    """

    if args.part in ("all", "report"):
        lineplots = ["insertion", "deletion", "infidelity", "max-sens", "s-impact", "impact", "sens-n"]
        boxplots = ["i-coverage", "del-until-flip"]

        # Create line plots
        output_file("report.html")
        figures = []
        for metric in lineplots:
            legend_data = []
            data = result_obj.get_metric(metric)
            if data:
                metadata = result_obj.metadata[metric]
                p = figure(title=metric, x_axis_label="x", y_axis_label="y", plot_width=1000, plot_height=500)
                col_iter = iter(palette)
                for method in data:
                    col = next(col_iter)
                    m_data = data[method]
                    if len(m_data.shape) > 1:
                        mean = np.mean(m_data, axis=0)
                        sd = np.std(m_data, axis=0)
                        l = p.line(x=result_obj.metadata[metric], y=mean, color=col, line_width=2)
                        a = p.varea(x=result_obj.metadata[metric],
                                    y1=(mean - (1.96 * sd / np.sqrt(m_data.shape[0]))),
                                    y2=(mean + (1.96 * sd / np.sqrt(m_data.shape[0]))),
                                    color=col, alpha=0.2)
                        legend_data.append((method, [l, a]))
                    else:
                        l = p.line(x=result_obj.metadata[metric], y=m_data, color=col, line_width=2)
                        legend_data.append((method, [l]))
                legend = Legend(items=legend_data)
                legend.click_policy = "hide"
                p.add_layout(legend, "right")
                figures.append(p)

        # Create box plots (impact coverage)
        # TODO check if this plotting code is correct (compare to matplotlib)
        renderer = hv.renderer("bokeh")
        for metric in boxplots:
            legend_data = []
            data = result_obj.get_metric(metric)
            if data:
                names = [method for method in data for _ in range(data[method].shape[0])]
                plot_data = np.concatenate([data[method] for method in data])
                boxwhisker = hv.BoxWhisker((names, plot_data), "Method", metric, label=metric)
                boxwhisker.opts(width=1000, height=500, xrotation=45)
                figures.append(renderer.get_plot(boxwhisker).state)

        show(column([Div(text=f"<h1>Directory: {args.dir}</h1>")] + figures))

    """
    PART 2: CONSISTENCY AND CORRELATIONS
    """
    if args.part in ("all", "consistency"):
        metrics = [m for m in result_obj.metric_names if m not in ("impact", "s-impact")]
        html_str = ""

        if args.bs_size > 0:
            result_obj.set_bootstrap(args.bs_size, args.bs_amt, 1024)
            if "i-coverage" in metrics:
                print("i-coverage not compatible with bootstrap, removing i-coverage")
                metrics.remove("i-coverage")

        # Calculate Krippendorff alpha for each metric (where applicable)
        k_a = {}
        for metric in metrics:
            if args.bs_size > 0:
                data = np.stack(
                    [result_obj.bootstrap(method, metric, columns=args.cols) for method in result_obj.method_names],
                    axis=1)
            else:
                data = np.stack(
                    [result_obj.aggregate(method, metric, columns=args.cols) for method in result_obj.method_names],
                    axis=1)
            k_a[metric] = krippendorff_alpha(np.argsort(data))
        table_str = "<table><tr><th>Method</th><th>Krippendorff Alpha</th></tr>"
        for key in k_a:
            table_str += f"<tr><td>{key}</td><td>{k_a[key]:.4f}</td></tr>"
        table_str += "</table>"
        html_str += table_str


        # Calculate inter-method reliability
        fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 10*len(metrics)))
        for i, metric in enumerate(metrics):
            if args.bs_size > 0:
                data = np.stack(
                    [result_obj.bootstrap(method, metric, columns=args.cols) for method in result_obj.method_names],
                    axis=0)
            else:
                data = np.stack(
                    [result_obj.aggregate(method, metric, columns=args.cols) for method in result_obj.method_names],
                    axis=0)
            print(metric, data.shape)
            corrs = spearmanr(data, axis=1)[0]
            correlation_heatmap(axs[i], corrs, result_obj.method_names, metric)
        html_str += convert_to_html(fig)

        # Calculate internal consistency
        # i-coverage doesn't necessarily have the same shape as the other metrics, so can't be compared properly
        icr_metrics = [m for m in metrics if m != "i-coverage"]
        fig, axs = plt.subplots(len(result_obj.method_names), 1, figsize=(10, 10*len(metrics)))
        for i, method in enumerate(result_obj.method_names):
            if args.bs_size > 0:
                data = np.stack(
                    [result_obj.bootstrap(method, metric, columns=args.cols) for metric in icr_metrics],
                    axis=0)
            else:
                data = np.stack(
                    [result_obj.aggregate(method, metric, columns=args.cols) for metric in icr_metrics],
                    axis=0)
            corrs = spearmanr(data, axis=1)[0]
            correlation_heatmap(axs[i], corrs, icr_metrics, method)
        html_str += convert_to_html(fig)

        with open("consistency.html", "w") as fp:
            fp.write(html_str)
            fp.close()
        webbrowser.open(f"file://{os.path.realpath('consistency.html')}", new=2)

    if args.part in ("all", "cluster"):
        metrics = [m for m in result_obj.metric_names if m not in ("impact", "s-impact")]
        html_str = ""
        for metric in metrics:
            data = np.stack([result_obj.aggregate(method, metric) for method in result_obj.method_names], axis=0)
            data = preprocessing.scale(data)
            Z = hierarchy.linkage(data, "single")
            fig, ax = plt.subplots()
            dn = hierarchy.dendrogram(Z, labels=result_obj.method_names, orientation="left", ax=ax)
            html_str += f"<h1>{metric}</h1>" + convert_to_html(fig)
        with open("cluster.html", "w") as fp:
            fp.write(html_str)
            fp.close()
        webbrowser.open(f"file://{os.path.realpath('cluster.html')}", new=2)

    if args.part in ("all", "global_cluster"):
        metrics = [m for m in result_obj.metric_names if m not in ("impact", "s-impact")]
        methods = [m for m in result_obj.method_names if m not in ("Random", "EdgeDetection")]
        html_str = ""
        data = []
        for metric in metrics:
            data.append(np.array([np.mean(result_obj.aggregate(method, metric)) for method in methods]))
        data = np.stack(data, axis=0).transpose()
        data = preprocessing.scale(data)
        Z = hierarchy.linkage(data, "single")
        fig, ax = plt.subplots()
        dn = hierarchy.dendrogram(Z, labels=np.array(methods), orientation="left", ax=ax)
        html_str += f"<h1>Cross-metric clustering</h1>" + convert_to_html(fig)
        with open("cluster_global.html", "w") as fp:
            fp.write(html_str)
            fp.close()
        webbrowser.open(f"file://{os.path.realpath('cluster_global.html')}", new=2)

    if args.part in ("all", "boxplots"):
        metrics = [m for m in result_obj.metric_names if m not in ("impact", "s-impact")]
        methods = result_obj.method_names
        html_str = ""
        for metric in metrics:
            data = [result_obj.aggregate(method, metric) for method in methods]
            order = np.argsort([np.median(d) for d in data])
            data = [data[o] for o in order]
            fig, ax = plt.subplots()
            ax.boxplot(data, showfliers=False)
            ax.set_title(metric)
            ax.set_xticklabels([methods[o] for o in order], rotation=45)
            html_str += convert_to_html(fig)
        with open("boxplots.html", "w") as fp:
            fp.write(html_str)
            fp.close()
        webbrowser.open(f"file://{os.path.realpath('boxplots.html')}", new=2)
