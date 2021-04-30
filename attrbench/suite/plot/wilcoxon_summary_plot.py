from attrbench.suite import SuiteResult


class WilcoxonSummaryPlot:
    def __init__(self, suite_result: SuiteResult):
        self.suite_result = suite_result

    def render(self, **kwargs):
        dfs = [m_res.get_df(**kwargs) for m_name, m_res in self.suite_result.metric_results.items()]
        return dfs


if __name__ == "__main__":
    res = SuiteResult.load_hdf("../../../out/imagenet_resnet18.h5")
    wsp = WilcoxonSummaryPlot(res)
    dfs = wsp.render()
