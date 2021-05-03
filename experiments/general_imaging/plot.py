from attrbench.suite import SuiteResult


if __name__ == "__main__":
    res = SuiteResult.load_hdf("../../out/imagenet_resnet18.h5")
    for key in res.metric_results:
        df, inverted = res.metric_results[key].get_df(mode="std_dist")
        df2, inverted2 = res.metric_results[key].get_df(mode="raw_dist")

    """
    dfs = {}
    for m_name, m_res in res.metric_results.items():
        dfs[m_name] = m_res.get_df()[0]
    """
