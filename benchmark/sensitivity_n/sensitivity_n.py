from typing import Iterable, Callable, List, Dict
import numpy as np


# Returns a dictionary containing, for each given method, a list of Sensitivity-n values
# where the values of n are given by mask_range
def sensitivity_n(data: Iterable, model: Callable[[np.ndarray], np.ndarray],
                  methods: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]], mask_range: List[int],
                  n_subsets=100, mask_value=0.):
    result = {m_name: [[] for _ in mask_range] for m_name in methods}
    for batch_index, (samples, labels) in enumerate(data):
        print(f"Batch {batch_index}...")
        sample_size = np.prod(samples.shape[1:])
        # Get original output and attributions
        orig_output = model(samples)
        attrs = {m_name: methods[m_name](samples, labels) for m_name in methods}
        for n_idx, n in enumerate(mask_range):
            output_diffs = []
            sum_of_attrs = {m_name: [] for m_name in methods}
            for _ in range(n_subsets):
                # Generate mask and masked samples
                # batch_dim: [batch_size, n] (made to match unravel_index output)
                mask = np.random.choice(sample_size, n)
                unraveled = np.unravel_index(mask, samples.shape[1:])
                batch_dim = np.array(list(range(samples.shape[0])) * n).reshape(-1, samples.shape[0]).transpose()
                masked_samples = np.copy(samples)
                masked_samples[(batch_dim, *unraveled)] = mask_value

                output = model(masked_samples)
                # Get difference in output confidence for desired class
                output_diffs.append((orig_output - output)[np.arange(samples.shape[0]), labels].reshape(samples.shape[0], 1))
                # Get sum of attributions of masked pixels
                for m_name in methods:
                    sum_of_attrs[m_name].append(
                        attrs[m_name][(batch_dim, *unraveled)]
                        .reshape(samples.shape[0], -1)
                        .sum(axis=1)
                        .reshape(samples.shape[0], 1))
            output_diffs = np.hstack(output_diffs)
            for m_name in methods:
                sum_of_attrs[m_name] = np.hstack(sum_of_attrs[m_name])
                result[m_name][n_idx] += [np.corrcoef(output_diffs[i], sum_of_attrs[m_name][i])[0, 1]
                                          for i in range(samples.shape[0])]
    for m_name in methods:
        result[m_name] = np.array(result[m_name])
    return result


if __name__ == "__main__":
    from util.vars import DATASET_MODELS
    from util.methods import get_method_constructors
    import itertools
    DATASET = "MNIST"
    DOWNLOAD_DATASET = False
    MODEL = "CNN"
    BATCH_SIZE = 64
    N_BATCHES = 2
    N_SUBSETS = 100
    #MASK_RANGE = range(1, 1000, 100)
    MASK_RANGE = range(1, 700, 50)
    METHODS = ["GuidedGradCAM", "Gradient", "InputXGradient", "IntegratedGradients",
               "GuidedBackprop", "Deconvolution", "Random"]

    # TODO instead of having this global dictionary, expose getter methods that do the necessary validations
    dataset_constructor = DATASET_MODELS[DATASET]["constructor"]
    model_constructor = DATASET_MODELS[DATASET]["models"][MODEL]
    method_constructors = get_method_constructors(METHODS)

    all_kwargs = {"Occlusion": {"sliding_window_shapes": (1, 1, 1)}}

    model = model_constructor(output_logits=True)
    dataset = dataset_constructor(batch_size=BATCH_SIZE, shuffle=False, download=DOWNLOAD_DATASET,
                                  data_location="../../../data")

    x = np.array(MASK_RANGE)
    methods = {m_name: method_constructors[m_name](model, normalize=True, **all_kwargs.get(m_name, {})) for m_name in METHODS}
    sensitivity_n(itertools.islice(dataset.get_test_loader(), N_BATCHES), lambda x: model.predict(x).detach().numpy(),
                  methods, list(MASK_RANGE), mask_value=-0.4242)
