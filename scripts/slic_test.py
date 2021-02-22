import matplotlib.pyplot as plt
import numpy as np

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

from experiments.general_imaging.dataset_models import get_dataset_model
from torch.utils.data import DataLoader
import torch

torch.manual_seed(1)
ds, model, patch_folder = get_dataset_model("MNIST")
dl = DataLoader(ds, batch_size=4, shuffle=True)
batch, labels = next(iter(dl))

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
ax = ax.ravel()
for i in range(batch.size(0)):
    img = batch[i, ...].numpy()
    if batch.size(1) == 3:
        img = img.transpose((1, 2, 0))
    else:
        img = img.squeeze()
    img = (img - img.min()) / (img.max() - img.min())
    #segments = slic(img, n_segments=10, compactness=10, sigma=1, start_label=1)
    segments = slic(img, n_segments=10, compactness=10, sigma=1, start_label=1, slic_zero=True)
    print(f"Number of segments {i}: {len(np.unique(segments))}")
    ax[i].imshow(mark_boundaries(img, segments))
    ax[i].set_axis_off()

plt.tight_layout()
plt.show()