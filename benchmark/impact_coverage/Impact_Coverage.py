import torch
import numpy as np
from sklearn.metrics import jaccard_score
import random
from util import datasets, models, methods
from os import path

DEVICE = "cuda"
DATASET = "CIFAR10"
DOWNLOAD_DATASET = False
BATCH_SIZE = 8
N_BATCHES = 32
N_SUBSETS = 100
TARGET_LABEL = 0
data_root = "../../../data"
normalize_attrs = True

PATCH_SIZE_PERCENT=0.1

patch = torch.load(path.join(data_root, "patches", "cifar10_resnet20_patch.pt"))
dataset = datasets.Cifar(batch_size=BATCH_SIZE, data_location=path.join(data_root, "CIFAR10"))
model = models.CifarResnet(version="resnet20", params_loc=path.join(data_root, "models/cifar10_resnet20.pth"),
                           output_logits=True)
model.to(DEVICE)
model.eval()

attribution_methods = {
    "GuidedGradCAM": methods.GuidedGradCAM(model, model.get_last_conv_layer(), normalize=normalize_attrs),
    "Gradient": methods.Gradient(model, normalize=normalize_attrs),
    "InputXGradient": methods.InputXGradient(model, normalize=normalize_attrs),
    "IntegratedGradients": methods.IntegratedGradients(model, normalize=normalize_attrs),
    "GuidedBackprop": methods.GuidedBackprop(model, normalize=normalize_attrs),
    "Deconvolution": methods.Deconvolution(model, normalize=normalize_attrs),
    "Random": methods.Random(normalize=normalize_attrs)
}


iterator = iter(dataset.get_test_data())
random.seed(1)

image_size = dataset.sample_shape[-1]
patch_size = int(((image_size**2)*PATCH_SIZE_PERCENT)**0.5)
keep_list = []  # boolean mask of images to keep (not predicted as target class and successfully attacked)
patch_masks = []  # boolean mask of location of patch
critical_factor_mask = {m_name: [] for m_name in attribution_methods}  # boolean mask of location top factors

for b in range(N_BATCHES):
    samples, labels = next(iterator)
    samples, labels = torch.tensor(samples), torch.tensor(labels)
    samples = samples.to(DEVICE, non_blocking=True)
    labels = labels.numpy()
    predictions = model(samples).detach().cpu().numpy()
    # patch location same for all images in batch
    ind = random.randint(0, image_size - patch_size)
    samples[:, :, ind:ind + patch_size, ind:ind + patch_size] = patch
    patch_location_mask = np.zeros(samples.shape)
    patch_location_mask[:,:, ind: ind + patch_size, ind: ind + patch_size] = 1.
    patch_masks.append(patch_location_mask)
    adv_out = model(samples).detach().cpu().numpy()
    # use images that are not of the targeted class and are successfully attacked
    keep_indices = (predictions.argmax(axis=1) != TARGET_LABEL) * (adv_out.argmax(axis=1) == TARGET_LABEL) * (labels != TARGET_LABEL)
    keep_list.extend(keep_indices)
    for m_name in critical_factor_mask:
        attr_samples = samples.clone()
        method = attribution_methods[m_name]
        attrs = method(samples, target=TARGET_LABEL)
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

for m_name in critical_factor_mask:
    cr_f_m = np.vstack(critical_factor_mask[m_name])
    cr_f_m = cr_f_m[keep_list]
    result[m_name] = jaccard_score(patch_masks.flatten(),cr_f_m.flatten())

print(result)
