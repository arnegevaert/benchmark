from attrbench.suite import SuiteResult
from attrbench.suite.plot import InterMethodCorrelationPlot, InterMetricCorrelationPlot
import matplotlib.pyplot as plt


def plot_wsp(res, mode):
    dfs = {metric_name: res.metric_results[metric_name].get_df(mode=mode, include_baseline=True) for metric_name in res.metric_results}
    wsp = InterMetricCorrelationPlot(
        {key: value[0] for key, value in dfs.items()},
        {key: value[1] for key, value in dfs.items()})
    fig = wsp.render(figsize=(10, 7), glyph_scale=1000)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    res = SuiteResult.load_hdf("../../out/imagenet_resnet18.h5")

    plot_wsp(res, mode="std_dist")
    plot_wsp(res, mode="raw_dist")
    plot_wsp(res, mode="raw")
