from methods import get_method_constructors
from vars import DATASET_MODELS
from lib import Report
import numpy as np
import torch


DATASET = "MNIST"
DOWNLOAD_DATASET = False
MODEL = "CNN"
BATCH_SIZE = 64
N_BATCHES = 16
MASK_RANGE = range(1, 128, 10)
N_EXAMPLES = 4
SAVE_REPORT = False
REPORT_LOC = "testreport"
METHODS = ["GuidedGradCAM", "Gradient", "InputXGradient", "IntegratedGradients",
           "GuidedBackprop", "Deconvolution", "Random"]

dataset_constructor = DATASET_MODELS[DATASET]["constructor"]
model_constructor = DATASET_MODELS[DATASET]["models"][MODEL]
# method_constructors = get_all_method_constructors()
method_constructors = get_method_constructors(METHODS)

all_kwargs = {"Occlusion": {"sliding_window_shapes": (1, 1, 1)}}

model = model_constructor()
dataset = dataset_constructor(batch_size=BATCH_SIZE, shuffle=False, download=DOWNLOAD_DATASET)

x = np.array(MASK_RANGE)
report = Report(list(method_constructors.keys()))
example_idxs = x[np.round(np.linspace(0, len(x) - 1, N_EXAMPLES)).astype(int)]
for key in method_constructors:
    method_examples = []
    print(f"Method: {key}")
    result = []
    iterator = iter(dataset.get_test_data())
    # Get any provided kwargs for this method
    kwargs = all_kwargs.get(key, {})
    method = method_constructors[key](model, **kwargs)
    for b in range(N_BATCHES):
        print(f"Batch {b+1}/{N_BATCHES}")
        samples, labels = next(iterator)
        batch_result = []
        attrs = method.attribute(samples, target=labels)  # [batch_size, *sample_shape]
        # Flatten each sample in order to sort indices per sample
        attrs = attrs.reshape(attrs.shape[0], -1)  # [batch_size, -1]
        # Sort indices of attrs in ascending order
        sorted_indices = attrs.argsort()  # [batch_size, -1]
        for i in MASK_RANGE:
            # Get indices of i most important inputs
            to_mask = sorted_indices[:, -i:]  # [batch_size, i]
            unraveled = np.unravel_index(to_mask, samples.shape[1:])
            # Mask i most important inputs
            # Batch_dim: [BATCH_SIZE, i] (made to match unravel_index output)
            batch_dim = np.array(list(range(BATCH_SIZE))*i).reshape(-1, BATCH_SIZE).transpose()
            samples[(batch_dim, *unraveled)] = dataset.mask_value
            # Get predictions for result
            batch_result.append(model.predict(samples).gather(1, labels.reshape(-1, 1)))
            if i in example_idxs and b == 0:
                method_examples.append(np.array(samples[0].detach().numpy()))
        batch_result = torch.cat(batch_result, 1)  # [batch_size, n_pixels]
        result.append(batch_result)
    result = torch.cat(result, 0).detach().numpy().mean(axis=0)  # [n_batches*batch_size, n_pixels]
    report.add_summary_line(x, result, label=key)
    report.add_method_example_row(key, method_examples)
report.render(x_label="Number of masked pixels", y_label="Difference in model output")

if SAVE_REPORT:
    report.save(REPORT_LOC)
