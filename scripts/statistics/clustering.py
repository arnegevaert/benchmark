import argparse
import pandas as pd
import matplotlib.pyplot as plt
from attrbench.suite import SuiteResult
import logging
from scripts.statistics.util import split_df
from scripts.statistics.df_extractor import DFExtractor
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("out_file", type=str, nargs='?')
    args = parser.parse_args()

    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S")

    # Constant parameters, might be moved to args if necessary
    EXCLUDE = ["Random_pos_only", "GradCAM_no_relu", "GuidedGradCAM_no_relu"]
    BASELINE = "Random"
    RES_OBJ = SuiteResult.load_hdf(args.file)
    plt.rcParams["figure.dpi"] = 140

    masker = "constant"
    activation = "linear"
    dfe = DFExtractor(RES_OBJ)
    df_dict = {}
    for mode in ("morf", "lerf"):
        for metric in ("Insertion", "Deletion", "IROF", "IIOF"):
            df_dict[f"{metric}-{mode}"] = dict(metric=f"masker_{masker}.{metric.lower()}",
                                               mode=mode, activation=activation)
    dfe.add_metrics(df_dict)
    dfe.add_metric("Deletion Until Flip", f"masker_{masker}.deletion_until_flip")

    infid_mode = "mse"
    dfe.add_metrics({
        "Infidelity-gaussian": dict(metric="infidelity_gaussian", mode=infid_mode, activation=activation),
        "Infidelity-seg": dict(metric="infidelity_seg", mode=infid_mode, activation=activation)
    }, log_transform=False)

    dfs = dfe.get_dfs()

    avg_values = {}
    for metric_name, (df, inverted) in dfs.items():
        df, baseline = split_df(df, BASELINE, EXCLUDE)
        avg_values[metric_name] = df.mean(axis=0) if not inverted else -df.mean(axis=0)
        #avg_values[metric_name] = df.mean(axis=0)

    df = pd.DataFrame.from_dict(avg_values)
    normalized = MinMaxScaler().fit_transform(df)
    normalized = pd.DataFrame(normalized, columns=df.columns, index=df.index)

    fig = sns.clustermap(normalized, figsize=(15, 10))
    plt.tight_layout()
    if args.out_file:
        fig.savefig(args.out_file)
    else:
        plt.show()
