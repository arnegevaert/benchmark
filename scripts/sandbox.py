from experiments.lib.attribution import NewExpectedGradients, ExpectedGradients
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
    eg = ExpectedGradients(model, ds, num_samples=125)
    new_eg = NewExpectedGradients(model, ds, num_samples=125)

    dl = DataLoader(ds, batch_size=1, shuffle=True)
    batch, labels = next(iter(dl))
    batch = batch.to("cuda")
    labels = labels.to("cuda")
    print("Running EG...")
    start_t = time()
    expl = eg(batch, labels).cpu()
    expl = torch.mean(expl, dim=1, keepdim=True).repeat(1, 3, 1, 1)
    end_t = time()
    print(f"{(end_t - start_t):.3f}")
    print("Running new EG...")
    start_t = time()
    new_expl = new_eg(batch, labels).cpu()
    new_expl = torch.mean(new_expl, dim=1, keepdim=True).repeat(1, 3, 1, 1)
    end_t = time()
    print(f"{(end_t - start_t):.3f}")

    imgs = (batch - batch.min()) / (batch.max() - batch.min())
    all = torch.cat([expl, new_expl, imgs.cpu()])

    plt.imshow(F.to_pil_image(make_grid(all)))
    plt.show()
    #plt.imshow(F.to_pil_image(make_grid(new_expl).cpu()))
