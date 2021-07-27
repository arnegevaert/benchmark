from attrbench.suite import SuiteResult
from experiments.general_imaging.plot.dfs import get_default_dfs
from os import path
import numpy as np
from scipy.special import softmax
import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    out_dir = "../../../out"
    params = [("ImageNet", "imagenet")]
    mode = "single"
    method = "spearman"
    use_softmax = True

    for ds_name, filename in params:
        hdf_name = filename + ".h5"
        csv_name = filename + ".csv"
        res_obj = SuiteResult.load_hdf(path.join(out_dir, hdf_name))
        dfs = get_default_dfs(res_obj, mode=mode)
        logits = np.loadtxt(path.join(out_dir, "confidence", csv_name), delimiter=",")
        if use_softmax:
            confidences = np.max(softmax(logits, axis=1), axis=1)
        else:
            confidences = np.max(logits, axis=1)

        corrs = {}
        for key, (df, inverted) in dfs.items():
            df = -df if inverted else df
            corrs[key] = df.corrwith(pd.Series(confidences), method=method)
        df = pd.DataFrame(corrs)

        sns.heatmap(df, annot=True, vmin=-1, vmax=1, cmap=sns.diverging_palette(220, 20, as_cmap=True))
