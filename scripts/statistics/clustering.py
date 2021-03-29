import argparse
import pandas as pd
import matplotlib.pyplot as plt
from attrbench.suite import SuiteResult
from os import path
import os
import logging
from scripts.statistics.util import get_dfs
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("out_file", type=str)
    args = parser.parse_args()

    # Constant parameters, might be moved to args if necessary
    IGNORE_METHODS = ["Random_pos_only", "GradCAM_no_relu", "GuidedGradCAM_no_relu"]
    BASELINE = "Random"
    PWR_ITERATIONS = 1000
    ALPHA = 0.01
    plt.rcParams["figure.dpi"] = 140
    res_obj = SuiteResult.load_hdf(args.file)

    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S")

    avg_values = {}
    for metric_name in res_obj.metric_results.keys():
        if "until_flip" not in metric_name:  # TODO handle del_until_flip as well
            dfs = get_dfs(res_obj, metric_name, BASELINE, IGNORE_METHODS)
            for variant, (df, baseline, inverted) in dfs.items():
                avg_values[f"{metric_name}_{variant}"] = df.mean(axis=0)
    df = pd.DataFrame.from_dict(avg_values)
    normalized = MinMaxScaler().fit_transform(df)
    normalized = pd.DataFrame(normalized, columns=df.columns, index=df.index)

    fig = sns.clustermap(normalized, figsize=(20, 15))
    fig.savefig(args.out_file)
