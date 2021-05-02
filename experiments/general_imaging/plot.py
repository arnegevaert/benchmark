from attrbench.suite import SuiteResult


if __name__ == "__main__":
    res = SuiteResult.load_hdf("../../out/imagenet_resnet18.h5")
    df, inverted = res.metric_results["impact_coverage"].get_df(mode="std_dist")

    """
    dfs = {}
    for m_name, m_res in res.metric_results.items():
        dfs[m_name] = m_res.get_df()[0]
    """
