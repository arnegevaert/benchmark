from scipy import stats
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from attrbench.suite import SuiteResult


def cohend(x, y, type="mean"):
    pooled_std = np.sqrt(((x.shape[0] - 1) * np.var(x) + (y.shape[0] - 1) * np.var(y)) / (x.shape[0] + y.shape[0] - 2))
    if type == "mean":
        return (np.mean(x) - np.mean(y)) / pooled_std
    elif type == "median":
        return (np.median(x) - np.median(y)) / pooled_std
    else:
        raise ValueError("Type must be mean or median")


def emp_power_curve(sample, baseline_sample, effect_size, iterations, n_range, inverted):
    emp_power = []
    for n in tqdm(n_range):
        bs_indices = np.random.choice(np.arange(sample.shape[0]), size=(iterations, n))
        bs_sample = sample[bs_indices]
        bs_baseline = baseline_sample[bs_indices]
        pvalues = [stats.wilcoxon(bs_sample[i, ...], bs_baseline[i, ...], alternative="less" if inverted else "greater")[1]
                   for i in range(iterations)]
        effect_sizes = [cohend(bs_sample[i, ...], bs_baseline[i, ...]) for i in range(iterations)]
        detected = (np.array(pvalues) < 0.01) & (np.array(effect_sizes) <= 0.9 * effect_size)
        emp_power.append(detected.sum() / iterations)
    return emp_power


def get_df(res_obj, name, baseline_method, mode=None, activation=None, ignore_methods=None):
    df_dict = res_obj.metric_results[name].to_df()
    df = df_dict[f"{mode}_{activation}"]
    inverted = res_obj.metric_results[name].inverted[mode]
    if ignore_methods is not None:
        df = df[df.columns.difference(ignore_methods)]
    baseline = df[baseline_method]
    df = df[df.columns.difference([baseline_method])]
    return df, baseline, inverted


if __name__ == "__main__":
    """
    full_sample = np.random.normal(loc=0.25, scale=1.0, size=10000)
    emp_power = emp_power_curve(full_sample, iterations=1000, n_range=np.arange(10, 1000, 10))
    plt.plot(np.arange(10, 1000, 10), emp_power)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("metric", type=str)
    parser.add_argument("method", type=str)
    parser.add_argument("-m", "--mode", type=str, default="mse")
    parser.add_argument("-a", "--activation", type=str, default="linear")
    args = parser.parse_args()

    IGNORE_METHODS = ["Random_pos_only", "GradCAM_no_relu", "GuidedGradCAM_no_relu"]
    BASELINE = "Random"
    PILOT_ROWS = 128
    ITERATIONS = 1000
    n_range = np.arange(1, 128, 5)
    plt.rcParams["figure.dpi"] = 140
    res_obj = SuiteResult.load_hdf(args.file)

    if args.metric == "infidelity":
        #infidelity_names = [f"infidelity_{pert}" for pert in ("gaussian", "seg", "sq")]
        infidelity_names = [f"infidelity_sq"]
        for name in infidelity_names:
            df, baseline, inverted = get_df(res_obj, name, BASELINE, args.mode, args.activation, IGNORE_METHODS)
            df_pilot = df.iloc[:PILOT_ROWS, :]
            df_new = df.iloc[PILOT_ROWS:, :]
            baseline_pilot = baseline.iloc[:PILOT_ROWS]
            baseline_new = baseline.iloc[PILOT_ROWS:]

            effect_size = cohend(df_pilot[args.method].to_numpy(), baseline_pilot.to_numpy())

            method_results = df_new[args.method].to_numpy()
            emp_power = emp_power_curve(method_results, baseline_new.to_numpy(), effect_size, ITERATIONS, n_range, inverted)
            print(emp_power)
            plt.plot(n_range, emp_power)
        plt.show()
