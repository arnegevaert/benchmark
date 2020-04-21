from typing import Iterable, Callable, List, Dict
import numpy as np


def simple_sensitivity(data: Iterable, model: Callable[[np.ndarray], np.ndarray],
                       methods: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],
                       mask_range: List[int], mask_value: float):
    result = {m_name: [] for m_name in methods}
    for batch_index, (samples, labels) in enumerate(data):
        for key in methods:
            batch_result = []
            attrs = methods[key](samples, labels)  # [batch_size, *sample_shape]
            # Flatten each sample in order to sort indices per sample
            attrs = attrs.reshape(attrs.shape[0], -1)  # [batch_size, -1]
            # Sort indices of attrs in ascending order
            sorted_indices = attrs.argsort()
            for i in mask_range:
                # Get indices of i most important inputs
                to_mask = sorted_indices[:, -i:]  # [batch-size, i]
                unraveled = np.unravel_index(to_mask, samples.shape[1:])
                # Mask i most important inputs
                # batch_dim: [batch_size, i] (made to match unravel_index output)
                batch_size = samples.shape[0]
                batch_dim = np.array(list(range(batch_size))*i).reshape(-1, batch_size).transpose()
                masked_samples = samples.clone()
                masked_samples[(batch_dim, *unraveled)] = mask_value
                # Get predictions for result
                predictions = model(masked_samples)
                predictions = predictions[np.arange(predictions.shape[0]), labels].reshape(-1, 1)
                batch_result.append(predictions)
            batch_result = np.concatenate(batch_result, axis=1)
            result[key].append(batch_result)
    for key in methods:
        result[key] = np.concatenate(result[key], axis=0).mean(axis=0)
    return result


if __name__ == '__main__':
    from util.vars import DATASET_MODELS
    from util.methods import get_method_constructors
    import itertools

    DATASET = "MNIST"
    MODEL = "CNN"
    BATCH_SIZE = 64
    N_BATCHES = 16
    MASK_RANGE = list(range(1, 128, 10))

    METHODS = ["GuidedGradCAM", "Gradient", "InputXGradient", "IntegratedGradients",
               "GuidedBackprop", "Deconvolution", "Random"]

    dataset_constructor = DATASET_MODELS[DATASET]["constructor"]
    model_constructor = DATASET_MODELS[DATASET]["models"][MODEL]
    # method_constructors = get_all_method_constructors()
    method_constructors = get_method_constructors(METHODS)

    model = model_constructor()
    dataset = dataset_constructor(batch_size=BATCH_SIZE, shuffle=False, data_location="../../data")
    methods = {m_name: method_constructors[m_name](model) for m_name in METHODS}

    result = simple_sensitivity(itertools.islice(iter(dataset.get_test_loader()), N_BATCHES),
                                lambda x: model.predict(x).detach().numpy(), methods, MASK_RANGE, -.4242)
