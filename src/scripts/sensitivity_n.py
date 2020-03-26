from methods import get_method_constructors
from vars import DATASET_MODELS
from lib import Report
import numpy as np
import torch

DATASET = "MNIST"
DOWNLOAD_DATASET = False
MODEL = "CNN"
BATCH_SIZE = 32
N_BATCHES = 2
N_SUBSETS = 100
MASK_RANGE = range(1, 600, 15)

# TODO instead of having this global dictionary, expose getter methods that do the necessary validations
dataset_constructor = DATASET_MODELS[DATASET]["constructor"]
model_constructor = DATASET_MODELS[DATASET]["models"][MODEL]
method_constructors = get_method_constructors(["Gradient", "InputXGradient", "IntegratedGradients"])

all_kwargs = {"Occlusion": {"sliding_window_shapes": (1, 1, 1)}}

model = model_constructor()
dataset = dataset_constructor(batch_size=BATCH_SIZE, shuffle=False, download=DOWNLOAD_DATASET)

x = np.array(MASK_RANGE)
report = Report(list(method_constructors.keys()))
# TODO this might go faster if we bring the method loop inside (generate samples and model output only once)
for key in method_constructors:
    print(f"Method: {key}")
    iterator = iter(dataset.get_test_data())
    kwargs = all_kwargs.get(key, {})
    method = method_constructors[key](model, **kwargs)
    all_corrs = []
    for b in range(N_BATCHES):
        print(f"Batch {b + 1}/{N_BATCHES}")
        samples, labels = next(iterator)
        sample_size = np.prod(samples.shape[1:])
        # Get original output and attributions
        orig_output = model.predict(samples, logits=True)
        attrs = method.attribute(samples, target=labels)
        corrs = []
        for n in MASK_RANGE:
            sum_diffs = []
            sum_attrs = []
            for _ in range(N_SUBSETS):
                # Generate mask and masked samples
                mask = np.random.choice(sample_size, n)
                unraveled = np.unravel_index(mask, samples.shape[1:])
                # Batch_dim: [BATCH_SIZE, n] (made to match unravel_index output)
                batch_dim = np.array(list(range(BATCH_SIZE)) * n).reshape(-1, BATCH_SIZE).transpose()
                masked_samples = samples.clone()
                masked_samples[(batch_dim, *unraveled)] = dataset.mask_value
                # Get model output of masked samples
                output = model.predict(masked_samples, logits=True)
                # Get sum of absolute difference in outputs ([BATCH_SIZE, 1])
                sum_diffs.append(
                    (orig_output - output).gather(1, labels.reshape(-1, 1)).reshape(BATCH_SIZE, -1).sum(dim=1).reshape(BATCH_SIZE, 1)
                        .detach().numpy())
                # Get sum of attributions ([BATCH_SIZE, 1])
                sum_attrs.append(
                    (attrs[(batch_dim, *unraveled)]).reshape(BATCH_SIZE, -1).sum(dim=1).reshape(BATCH_SIZE, 1)
                        .detach().numpy())
            sum_diffs = np.hstack(sum_diffs)
            sum_attrs = np.hstack(sum_attrs)
            # Get average correlation between diff in output and sum of attributions for this value of n
            corrs.append(np.average([np.corrcoef(sum_diffs[i], sum_attrs[i])[0, 1] for i in range(BATCH_SIZE)]))
        all_corrs.append(corrs)  # corrs has shape [len(MASK_RANGE)]
    # all_corrs has shape [N_BATCHES, len(MASK_RANGE)]
    result = np.average(all_corrs, axis=0)
    report.add_summary_line(x, result, label=key)
report.render()
