from methods import get_method_constructors
from vars import DATASET_MODELS
from lib import Report
import numpy as np
import time

DATASET = "CIFAR10"
DOWNLOAD_DATASET = False
MODEL = "resnet20"
BATCH_SIZE = 64
N_BATCHES = 16
N_SUBSETS = 100
MASK_RANGE = range(1, 1000, 100)
#MASK_RANGE = range(1, 700, 50)
METHODS = ["GuidedGradCAM", "Gradient", "InputXGradient", "IntegratedGradients",
           "GuidedBackprop", "Deconvolution", "Random"]

# TODO instead of having this global dictionary, expose getter methods that do the necessary validations
dataset_constructor = DATASET_MODELS[DATASET]["constructor"]
model_constructor = DATASET_MODELS[DATASET]["models"][MODEL]
method_constructors = get_method_constructors(METHODS)

all_kwargs = {"Occlusion": {"sliding_window_shapes": (1, 1, 1)}}

model = model_constructor(output_logits=True)
dataset = dataset_constructor(batch_size=BATCH_SIZE, shuffle=False, download=DOWNLOAD_DATASET)

x = np.array(MASK_RANGE)
report = Report(list(method_constructors.keys()))
methods = {m_name: method_constructors[m_name](model, normalize=True, **all_kwargs.get(m_name, {})) for m_name in METHODS}
iterator = iter(dataset.get_test_data())
method_corrs = {m_name: [] for m_name in METHODS}

all_corrs = {m_name: [] for m_name in METHODS}
# TODO put time prediction code in class to plug-in easily
for b in range(N_BATCHES):
    start_t = time.time()
    print(f"Batch {b + 1}/{N_BATCHES}...")
    samples, labels = next(iterator)
    sample_size = np.prod(samples.shape[1:])
    # Get original output and attributions
    orig_output = model.predict(samples)
    attrs = {m_name: methods[m_name].attribute(samples, target=labels) for m_name in METHODS}
    corrs = {m_name: [] for m_name in METHODS}
    for n in MASK_RANGE:
        sum_diffs = []
        sum_attrs = {m_name: [] for m_name in METHODS}
        for _ in range(N_SUBSETS):
            # Generate mask and masked samples
            mask = np.random.choice(sample_size, n)
            unraveled = np.unravel_index(mask, samples.shape[1:])
            # Batch_dim: [BATCH_SIZE, n] (made to match unravel_index output)
            batch_dim = np.array(list(range(BATCH_SIZE)) * n).reshape(-1, BATCH_SIZE).transpose()
            masked_samples = samples.clone()
            masked_samples[(batch_dim, *unraveled)] = dataset.mask_value
            # Get model output of masked samples
            output = model.predict(masked_samples)
            # Get sum of absolute difference in outputs ([BATCH_SIZE, 1])
            sum_diffs.append(
                (orig_output - output).gather(1, labels.reshape(-1, 1)).reshape(BATCH_SIZE, -1).sum(dim=1).reshape(BATCH_SIZE, 1)
                    .detach().numpy())
            # Get sum of attributions for each method ([BATCH_SIZE, 1])
            for m_name in METHODS:
                sum_attrs[m_name].append(
                    (attrs[m_name][(batch_dim, *unraveled)]).reshape(BATCH_SIZE, -1).sum(dim=1).reshape(BATCH_SIZE, 1)
                        .detach().numpy())
        sum_diffs = np.hstack(sum_diffs)
        for m_name in METHODS:
            sum_attrs[m_name] = np.hstack(sum_attrs[m_name])
            # Get average correlation between diff in output and sum of attributions for this value of n
            corrs[m_name].append(np.average([np.corrcoef(sum_diffs[i], sum_attrs[m_name][i])[0, 1] for i in range(BATCH_SIZE)]))
    for m_name in METHODS:
        all_corrs[m_name].append(corrs[m_name])  # corrs has shape [len(MASK_RANGE)]
    end_t = time.time()
    seconds = end_t - start_t
    print(f"Batch {b+1}/{N_BATCHES} took {seconds:.2f}s. ETA: {seconds * (N_BATCHES-b-1):.2f}s.")

for m_name in METHODS:
    # all_corrs has shape [N_BATCHES, len(MASK_RANGE)]
    result = np.average(all_corrs[m_name], axis=0)
    report.add_summary_line(x, result, label=m_name)
report.render(x_label="Number of masked pixels", y_label="Pearson Correlation between output change and attribution",
              y_range=(-.1, 1))
