import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic
from experiments.general_imaging.dataset_models import get_dataset_model
from torch.utils.data import DataLoader
from tqdm import tqdm

ds, model, patch_folder = get_dataset_model("ImageNette")
dl = DataLoader(ds, batch_size=32, shuffle=True)

n_segments = []
for batch, labels in tqdm(dl):
    for i in range(batch.size(0)):
        img = batch[i, ...].numpy()
        if batch.size(1) == 3:
            img = img.transpose((1, 2, 0))
        else:
            img = img.squeeze()
        img = (img - img.min()) / (img.max() - img.min())
        #segments = slic(img, n_segments=100, compactness=10, sigma=1, start_label=1)
        segments = slic(img, n_segments=100, start_label=1, slic_zero=True)
        n_segments.append(len(np.unique(segments)))

plt.hist(n_segments, bins=20)
print(f"min: {np.min(n_segments)}, max: {np.max(n_segments)}")

plt.tight_layout()
plt.show()
