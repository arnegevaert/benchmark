from noise_invariance.noise_perturbed_dataset import NoisePerturbedDataset
from typing import Callable, Dict
import numpy as np


def noise_invariance(data: NoisePerturbedDataset, methods: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]):
    result = {}
    for m_name in methods:
        method = methods[m_name]
        print(f"Method: {m_name}...")
        diffs = [[] for _ in range(len(data.perturbation_levels))]
        cur_max_diff = 0
        cur_max_diff_examples = {}
        for batch_idx, batch in enumerate(perturbed_dataset):
            print(f"Batch {batch_idx+1}/{data.n_batches}")
            orig = batch["original"]
            labels = batch["labels"]
            orig_attr = method(orig, labels)  # [batch_size, *sample_shape]
            for n_l, noise_level_batch in enumerate(batch["perturbed"]):
                perturbed_attr = method(noise_level_batch, labels)  # [batch_size, *sample_shape]
                avg_diff_per_image = np.average(np.reshape(np.abs(orig_attr - perturbed_attr), (data.batch_size, -1)),
                                                axis=1)  # [batch_size]
                max_diff_idx = np.argmax(avg_diff_per_image).item()
                if avg_diff_per_image[max_diff_idx] > cur_max_diff:
                    cur_max_diff = avg_diff_per_image[max_diff_idx]
                    cur_max_diff_examples = {
                        "orig": orig[max_diff_idx], "perturbed": noise_level_batch[max_diff_idx],
                        "orig_attr": orig_attr[max_diff_idx], "perturbed_attr": perturbed_attr[max_diff_idx],
                        "noise_level": data.perturbation_levels[n_l]
                    }
        result[m_name] = {
            "diffs": np.array(diffs),  # [noise_levels, n_batches]
            "max_diff": cur_max_diff_examples,
            "max_diff_exs": cur_max_diff_examples
        }
    return result


if __name__ == "__main__":
    from noise_invariance.noise_perturbed_dataset import NoisePerturbedDataset, generate_noise_perturbed_dataset
    from util.vars import DATASET_MODELS
    from util.methods import get_method_constructors
    import numpy as np
    from os import path

    GENERATE = False
    DATA_ROOT = "../../../data"
    DATASET = "CIFAR10"
    MODEL = "resnet20"
    BATCH_SIZE = 4
    N_BATCHES = 256
    METHODS = ["GuidedGradCAM", "Gradient", "InputXGradient", "IntegratedGradients",
               "GuidedBackprop", "Deconvolution"]

    dataset_name = f"{DATASET}_noise"
    dataset_constructor = DATASET_MODELS[DATASET]["constructor"]
    dataset = dataset_constructor(batch_size=BATCH_SIZE, download=False, shuffle=True,
                                  data_location=path.join(DATA_ROOT, DATASET))
    model_constructor = DATASET_MODELS[DATASET]["models"][MODEL]
    model = model_constructor()

    all_kwargs = {"Occlusion": {"sliding_window_shapes": (1, 1, 1)}}
    method_constructors = get_method_constructors(METHODS)
    methods = {m_name: method_constructors[m_name](model, normalize=True, **all_kwargs.get(m_name, {}))
               for m_name in METHODS}

    if GENERATE:
        perturbed_dataset = \
            generate_noise_perturbed_dataset(dataset.get_test_loader_numpy(),
                                             path.join(DATA_ROOT, dataset_name),
                                             perturbation_levels=list(np.linspace(0, 0.1, 10)), n_batches=N_BATCHES,
                                             model=lambda x: model.predict(x).detach().numpy(), max_tries=2)
    else:
        perturbed_dataset = NoisePerturbedDataset(path.join(DATA_ROOT, dataset_name))

    result = noise_invariance(perturbed_dataset, methods)
