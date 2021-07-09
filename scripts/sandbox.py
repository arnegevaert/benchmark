from experiments.lib.attribution import DeepShap, NewDeepShap
from experiments.general_imaging.lib.dataset_models import get_dataset_model
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import torch
from time import time


if __name__ == "__main__":
    ds, model, patch_folder = get_dataset_model("ImageNet", model_name="resnet18")
    model.eval()
    model.cuda()
    deepshap = DeepShap(model, ds, n_baseline_samples=10)
    newdeepshap = NewDeepShap(model, ds, n_baseline_samples=10)

    dl = DataLoader(ds, batch_size=2, shuffle=True)
    batch, labels = next(iter(dl))
    batch = batch.to("cuda")
    labels = labels.to("cuda")
    print("Running DeepSHAP...")
    start_t = time()
    expl = deepshap(batch, labels).cpu()
    expl = torch.mean(expl, dim=1, keepdim=True)
    end_t = time()
    print(f"{end_t - start_t}:.2f")
    print("Running new DeepSHAP...")
    start_t = time()
    new_expl = newdeepshap(batch, labels).cpu()
    new_expl = torch.mean(new_expl, dim=1, keepdim=True)
    end_t = time()
    print(f"{end_t - start_t}:.2f")

    both = torch.cat([expl, new_expl])

    plt.imshow(F.to_pil_image(make_grid(both, nrow=4)))
    plt.show()
    #plt.imshow(F.to_pil_image(make_grid(new_expl).cpu()))
