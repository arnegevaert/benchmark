from methods import get_method_constructors
from vars import DATASET_MODELS
import torch
from lib import Report
import numpy as np
import time
from sklearn.metrics import jaccard_score
import random

DEVICE = "cuda"
DATASET = "CIFAR10"
DOWNLOAD_DATASET = False
MODEL = "resnet20"
BATCH_SIZE = 64
N_BATCHES = 16
N_SUBSETS = 100
TARGET_LABEL = 0
MASK_RANGE = range(1, 500, 50)
MASK_RANGE = range(1, 700, 50)
METHODS = ["GuidedGradCAM", "Gradient", "InputXGradient", "IntegratedGradients",
           "GuidedBackprop", "Deconvolution", "Random"]

PATCH_PATH = "../models/saved_models/saved_patches/patch_checkpoint.pt"
PATCH_SIZE_PERCENT=0.1

patch = torch.load(PATCH_PATH)
dataset_constructor = DATASET_MODELS[DATASET]["constructor"]
model_constructor = DATASET_MODELS[DATASET]["models"][MODEL]
method_constructors = get_method_constructors(METHODS)

all_kwargs = {"Occlusion": {"sliding_window_shapes": (1, 1, 1)}}

model = model_constructor(output_logits=True)
model.to(DEVICE)
methods = {m_name: method_constructors[m_name](model, normalize=True, **all_kwargs.get(m_name, {})) for m_name in
           METHODS}
dataset = dataset_constructor(batch_size=BATCH_SIZE, shuffle=False, download=DOWNLOAD_DATASET)
iterator = iter(dataset.get_test_data())
random.seed(1)

image_size =dataset.get_sample_shape()[-1]
patch_size = int(((image_size**2)*PATCH_SIZE_PERCENT)**0.5)
keep_list = [] # boolean mask of images to keep (not predicted as target class and successfully attacked)
patch_masks =[] # boolean mask of location of patch
critical_factor_mask = {m_name: [] for m_name in METHODS} # boolean mask of location top factors

for b in range(N_BATCHES):
    samples, labels = next(iterator)
    samples = samples.to(DEVICE, non_blocking=True)
    labels = labels.numpy()
    predictions = model.predict(samples).detach().cpu().numpy()
    # patch location same for all images in batch
    ind = random.randint(0, image_size - patch_size)
    samples[:, :, ind:ind + patch_size, ind:ind + patch_size] = patch
    patch_location_mask = np.zeros(samples.shape)
    patch_location_mask[:,:, ind: ind + patch_size, ind: ind + patch_size] = 1.
    patch_masks.append(patch_location_mask)
    adv_out = model.predict(samples).detach().cpu().numpy()
    # use images that are not of the targeted class and are successfully attacked
    keep_indices = (predictions.argmax(axis=1) != TARGET_LABEL) * (adv_out.argmax(axis=1) == TARGET_LABEL) * (labels != TARGET_LABEL)
    keep_list.extend(keep_indices)
    for m_name in critical_factor_mask:
        attr_samples = samples.clone()
        method = methods[m_name]
        attrs = method.attribute(samples, target=TARGET_LABEL)
        attrs_shape = attrs.shape
        assert(len(attrs_shape) == 4) # shape[1] = 3 if colour channels are separate attributes, =1 pixel is attribute. might change later?
        attrs = attrs.reshape(attrs.shape[0], -1)
        sorted_indices = attrs.argsort().cpu()
        nr_top_attributes = attrs_shape[1]*patch_size**2
        to_mask = sorted_indices[:, -nr_top_attributes:]  # [batch_size, i]
        unraveled = np.unravel_index(to_mask, samples.shape[1:])
        batch_dim = np.column_stack([range(BATCH_SIZE) for i in range(nr_top_attributes)])
        masks = np.zeros(attr_samples.shape)
        masks[(batch_dim, *unraveled)] = 1.
        critical_factor_mask[m_name].append(masks)
result = {}
patch_masks = np.vstack(patch_masks)[keep_list]

report = Report(list(method_constructors.keys()))

for m_name in critical_factor_mask:
    cr_f_m = np.vstack(critical_factor_mask[m_name])
    cr_f_m = cr_f_m[keep_list]
    result[m_name] = jaccard_score(patch_masks.flatten(),cr_f_m.flatten())

print(result)
