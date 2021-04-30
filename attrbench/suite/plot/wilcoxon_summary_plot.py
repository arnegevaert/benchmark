from attrbench.suite import SuiteResult


class WilcoxonSummaryPlot:
    def __init__(self):
        pass

    def render(self, **kwargs):
        pass


if __name__ == "__main__":
    res = SuiteResult.load_hdf("../../../out/imagenet_resnet18.h5")
    dfs = {}
    for metric_name, metric_result in res.metric_results.items():
        dfs[metric_name] = metric_result.get_df()
