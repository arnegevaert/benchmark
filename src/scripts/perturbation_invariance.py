from datasets import PerturbedImageDataset
from vars import DATASET_MODELS
from methods import *
import numpy as np
import torch
from bokeh import plotting, layouts, palettes, models
import itertools

GENERATE = False
DATA_ROOT = "../../data"
DATASET = "MNIST"
PERT_FN = "mean_shift"
MODEL = "CNN"
BATCH_SIZE = 4
N_BATCHES = 16  # 128

dataset_name = f"{DATASET}_{PERT_FN}"
dataset_constructor = DATASET_MODELS[DATASET]["constructor"]
model_constructor = DATASET_MODELS[DATASET]["models"][MODEL]
model = model_constructor()

# method_constructors = get_all_method_constructors(include_random=False)
method_constructors = get_method_constructors(["Gradient", "InputXGradient", "GuidedGradCAM"])

if GENERATE:
    dataset = dataset_constructor(batch_size=BATCH_SIZE, download=False, shuffle=True)
    iterator = iter(dataset.get_test_data())
    perturbed_dataset = PerturbedImageDataset.generate(DATA_ROOT, dataset_name, iterator, model,
                                                       perturbation_fn=PERT_FN,
                                                       perturbation_levels=np.linspace(-1, 1, 10),
                                                       n_batches=N_BATCHES)
else:
    perturbed_dataset = PerturbedImageDataset(DATA_ROOT, dataset_name, BATCH_SIZE)


def plot_image(img, width=200, height=200, dw=1, dh=1):
    if type(img) == torch.Tensor:
        img = img.detach().numpy()

    is_grayscale = img.shape[0] == 1
    # Convert range to 0..255 uint32
    img = (img - np.min(img))/(np.max(img) - np.min(img))
    img *= 255
    img = np.array(img, dtype=np.uint32)

    # If grayscale, just provide rows/cols. If RGBA, convert to RGBA
    if is_grayscale:
        img = np.squeeze(img)
    else:
        img = img.transpose((1, 2, 0))
        rgba_img = np.empty(shape=(img.shape[0], img.shape[1], 4), dtype=np.uint8)
        rgba_img[:, :, :3] = img
        rgba_img[:, :, 3] = 255
        img = rgba_img
    img = np.flip(img, axis=0)
    p = plotting.figure(width=width, height=height)
    p.toolbar_location = None
    p.axis.visible = False
    p.grid.visible = False
    plot_fn = p.image if is_grayscale else p.image_rgba
    plot_fn([img], x=0, y=0, dw=dw, dh=dh)
    return p


result_plot = plotting.figure()
colors = itertools.cycle(palettes.Dark2_5)
imgplots = []
for key in method_constructors:
    print(f"Calculating for {key}...")
    method = method_constructors[key](model)
    diffs = [[] for _ in range(len(perturbed_dataset.get_levels()))]
    cur_max_diff = 0
    cur_max_diff_imgs = {
        "original": None, "perturbed": None, "orig_attr": None, "perturbed_attr": None
    }
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
                cur_max_diff_imgs = {
                    "original": orig[max_diff_idx], "perturbed": noise_level_batch[max_diff_idx],
                    "orig_attr": orig_attr[max_diff_idx], "perturbed_attr": perturbed_attr[max_diff_idx]
                }
            avg_diff = np.average(np.abs(orig_attr - perturbed_attr))
            diffs[n_l].append(avg_diff)
    diffs = np.array(diffs)
    avg_diffs = np.average(diffs, axis=1)

    result_plot.line(perturbed_dataset.get_levels(), diffs.mean(axis=1), legend_label=key,
                     color=next(colors), line_width=3)

    imgplots.append([models.Div(text=f"<h1>{key}</h1>", sizing_mode="stretch_width")])
    stacked_imgs = np.concatenate([cur_max_diff_imgs["original"], cur_max_diff_imgs["perturbed"]], axis=-1)
    stacked_imgs = np.clip(stacked_imgs, a_min=cur_max_diff_imgs["original"].min().item(), a_max=cur_max_diff_imgs["original"].max().item())
    imgplots.append([plot_image(stacked_imgs, width=400, dw=2),
                    plot_image(cur_max_diff_imgs["orig_attr"]), plot_image(cur_max_diff_imgs["perturbed_attr"])])
plotting.show(layouts.layout([[result_plot]] + imgplots))
