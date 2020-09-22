import torch
import os
from os import path
from torch.utils.data import DataLoader
from tqdm import tqdm

# This block allows us to import from the benchmark folder,
# as if it was a package installed using pip
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from experiments.lib import models
from bam import BAMDataset


if __name__ == "__main__":
    data_root = "../data"
    device = "cuda"

    for dataset in ["obj", "scene", "scene_only"]:
        chkp_dir = path.join(data_root, "models", "BAM", dataset, "resnet18")
        for checkpoint in os.listdir(chkp_dir):
            print(path.join(chkp_dir, checkpoint), dataset)
            model = models.Resnet("resnet18", output_logits=True, num_classes=10)
            model.load_state_dict(torch.load(path.join(chkp_dir, checkpoint)))
            model.to(device)

            train_ds = BAMDataset(path.join(data_root, "bam"), include_orig_scene=True, train=True)
            test_ds = BAMDataset(path.join(data_root, "bam"), include_orig_scene=True, train=False)

            train_dl = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4)
            test_dl = DataLoader(test_ds, batch_size=512, shuffle=True, num_workers=4)

            with torch.no_grad():
                for dl, desc in [(train_dl, "Train set"), (test_dl, "Test set")]:
                    total_samples, correct_samples = 0, 0
                    prog = tqdm(dl, desc=desc)
                    for overlay, orig_scene, scene_labels, object_labels in prog:
                        batch = overlay.to(device) if dataset != "scene_only" else orig_scene.to(device)
                        labels = object_labels.to(device) if dataset == "obj" else scene_labels.to(device)
                        y_pred = model(batch)
                        total_samples += batch.size(0)
                        correct_samples += (torch.argmax(y_pred, dim=1) == labels).sum().item()
                        prog.set_postfix({"acc": f"{correct_samples}/{total_samples} ({100 * correct_samples / total_samples:.2f}%)"})
