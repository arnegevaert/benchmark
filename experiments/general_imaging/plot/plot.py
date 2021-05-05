from attrbench.suite import SuiteResult
from attrbench.suite.plot import *
import matplotlib.pyplot as plt
from itertools import product


if __name__ == "__main__":
    #plt.ioff()
    res = SuiteResult.load_hdf("../../../out/imagenet_resnet18.h5")
    for mode in ("single_dist", "std_dist"):
        m_res = res.metric_results["deletion"]
        """
        dfs = {
            f"{afn} - {masker}": m_res.get_df(mode=mode, activation_fn=afn, masker=masker)
            for afn, masker in product(("linear", "softmax"), ("constant", "random", "blur"))
        }
        """
        df = m_res.get_df(mode=mode)[0]
        plot = ConvergencePlot(df)
        fig = plot.render(title=mode)
    #fig.tight_layout()
    #plt.show()
