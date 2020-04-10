from datasets import NoisePerturbedDataset
from vars import DATASET_MODELS
from methods import get_method_constructors
from lib import Report
import numpy as np
import torch

GENERATE = False
DATA_ROOT = "../../data"
DATASET = "CIFAR10"
PERT_FN = "noise"
MODEL = "resnet20"
BATCH_SIZE = 4
N_BATCHES = 256
METHODS = ["GuidedGradCAM", "Gradient", "InputXGradient", "IntegratedGradients",
           "GuidedBackprop", "Deconvolution"]

dataset_name = f"{DATASET}_{PERT_FN}"
dataset_constructor = DATASET_MODELS[DATASET]["constructor"]
model_constructor = DATASET_MODELS[DATASET]["models"][MODEL]
model = model_constructor()

all_kwargs = {"Occlusion": {"sliding_window_shapes": (1, 1, 1)}}
method_constructors = get_method_constructors(METHODS)

if GENERATE:
    dataset = dataset_constructor(batch_size=BATCH_SIZE, download=False, shuffle=True)
    perturbed_dataset = NoisePerturbedDataset.generate(DATA_ROOT, dataset_name, dataset, model,
                                                       perturbation_fn=PERT_FN,
                                                       perturbation_levels=np.linspace(0, 0.25, 10),  # np.linspace(0, 1, 10)
                                                       n_batches=N_BATCHES)
else:
    perturbed_dataset = NoisePerturbedDataset(DATA_ROOT, dataset_name, BATCH_SIZE)

report = Report(list(method_constructors.keys()))
for key in method_constructors:
    print(f"Calculating for {key}...")
    kwargs = all_kwargs.get(key, {})
    method = method_constructors[key](model, **kwargs)
    diffs = [[] for _ in range(len(perturbed_dataset.get_levels()))]
    cur_max_diff = 0
    for b, b_dict in enumerate(perturbed_dataset):
        print(f"Batch {b+1}/{N_BATCHES}")
        orig = torch.tensor(b_dict["original"])  # [batch_size, *sample_shape]
        labels = torch.tensor(b_dict["labels"])  # [batch_size]
        orig_attr = method.attribute(orig, target=labels).detach()  # [batch_size, *sample_shape]
        for n_l, noise_level_batch in enumerate(b_dict["perturbed"]):
            noise_level_batch = torch.tensor(noise_level_batch)  # [batch_size, *sample_shape]
            perturbed_attr = method.attribute(noise_level_batch, target=labels).detach()  # [batch_size, *sample_shape]

            avg_diff_per_image = np.average(
                np.reshape(
                    np.abs(orig_attr - perturbed_attr), (perturbed_dataset.batch_size, -1)
                ), axis=1
            )  # [batch_size]
            max_diff_idx = np.argmax(avg_diff_per_image).item()
            if avg_diff_per_image[max_diff_idx] > cur_max_diff:
                cur_max_diff = avg_diff_per_image[max_diff_idx]
                report.reset_method_examples(key)
                report.add_method_example_row(key, [orig[max_diff_idx], noise_level_batch[max_diff_idx]])
                report.add_method_example_row(key, [orig_attr[max_diff_idx], perturbed_attr[max_diff_idx]])
            diffs[n_l].append(np.average(avg_diff_per_image))
    diffs = np.array(diffs)  # [noise_levels, n_batches]
    report.add_summary_line(perturbed_dataset.get_levels(), diffs.mean(axis=1), label=key)
report.render(x_label="Noise level", y_label="Average attribution difference")

import os
report.save(os.path.join(__file__, "../../../out/cifar_perturbation_invariance.pkl"))