import argparse
from attrbench.suite import SuiteResult
from scripts.statistics._metric_scores import metric_scores
from attrbench import metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", type=str)
    args = parser.parse_args()
    res = SuiteResult.load_hdf(args.result_file)

    # 1) summarize distributions of metric scores and perform Wilcoxon signed rank tests
    dfs = {}
    for metric_name in res.metric_results:
        mr = res.metric_results[metric_name]
        if type(mr) == metrics.DeletionResult:
            dfs[metric_name] = mr.to_df()
