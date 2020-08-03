from typing import Iterable, Callable, Dict
from tqdm import tqdm
import torch
import numpy as np


def input_dependence_rate(dataloader: Iterable, model: Callable, methods: Dict[str, Callable],
                          device: str):
    """
    Input dependence rate:
    Percentage of correctly classified inputs with object overlay by model trained on scene labels,
    where region of object overlay is attributed less than that same region when the overlay
    is removed. 1 - IDR can be interpreted as "false positive rate": rate of explanations
    that assign higher importance to less important features.
    """
    result = {m_name: {"overlay": [], "scene": []} for m_name in methods}
    for images, scenes, masks, scene_labels, object_labels in tqdm(dataloader):
        images = images.to(device)
        scenes = scenes.to(device)
        labels = scene_labels.to(device)
        masks = masks.to(device)  # [batch_size, 1, height, width]
        with torch.no_grad():
            y_pred = torch.argmax(model(images), dim=1)
        # Boolean array indicating correctly classified images
        correctly_classified = (y_pred == labels)
        for m_name in methods:
            attrs_overlay = methods[m_name](images, labels)  # [batch_size, *sample_shape]
            attrs_scene = methods[m_name](scenes, labels)  # [batch_size, *sample_shape]
            masked_attrs_overlay = masks * attrs_overlay
            masked_attrs_scene = masks * attrs_scene
            mask_size = torch.sum(masks.flatten(1), dim=1)  # [batch_size]
            avg_attrs_overlay = masked_attrs_overlay.flatten(1).sum(dim=1) / mask_size
            avg_attrs_scene = masked_attrs_scene.flatten(1).sum(dim=1) / mask_size
            result[m_name]["overlay"].append(avg_attrs_overlay[correctly_classified].cpu().detach().numpy())
            result[m_name]["scene"].append(avg_attrs_scene[correctly_classified].cpu().detach().numpy())
    return {m_name: {"overlay": np.concatenate(result[m_name]["overlay"]),
                     "scene": np.concatenate(result[m_name]["scene"])}
            for m_name in methods}