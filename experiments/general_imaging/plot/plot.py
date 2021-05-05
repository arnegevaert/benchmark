from attrbench.suite import SuiteResult
from attrbench.suite.plot import InterMethodCorrelationPlot, InterMetricCorrelationPlot, WilcoxonSummaryPlot
import matplotlib.pyplot as plt
from itertools import product


if __name__ == "__main__":
    #plt.ioff()
    res = SuiteResult.load_hdf("../../../out/fashionmnist.h5")
    for mode in ("raw", "single_dist", "median_dist", "std_dist"):
        m_res = res.metric_results["sensitivity_n"]
        dfs = {
            f"{afn} - {masker}": m_res.get_df(mode=mode, activation_fn=afn, masker=masker)
            for afn, masker in product(("linear", "softmax"), ("constant", "random", "blur"))
        }
        plot = WilcoxonSummaryPlot(dfs)
        fig = plot.render(figsize=(10, 8), glyph_scale=1000, title=mode)
        fig.tight_layout()
    plt.show()
