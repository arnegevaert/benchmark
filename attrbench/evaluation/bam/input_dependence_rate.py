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
    result = {m_name: {"object_attrs": [], "scene_attrs": [], "mask_size": []} for m_name in methods}
    for images, scenes, masks, scene_labels, object_labels in tqdm(dataloader):
        images = images.to(device)
        scenes = scenes.to(device)
        labels = scene_labels.to(device)
        masks = masks.squeeze().to(device)  # [batch_size, 1, height, width]
        with torch.no_grad():
            y_pred = torch.argmax(model(images), dim=1)
        # Boolean array indicating correctly classified images
        correctly_classified = (y_pred == labels)
        for m_name in methods:
            object_attrs = methods[m_name](images, labels)  # [batch_size, *sample_shape]
            scene_attrs = methods[m_name](scenes, labels)  # [batch_size, *sample_shape]
            mask_size = torch.sum(masks.flatten(1), dim=1)  # [batch_size]
            masked_object_attrs = (masks * object_attrs).flatten(1).sum(dim=1)
            result[m_name]["object_attrs"].append(masked_object_attrs[correctly_classified].cpu().detach().numpy())
            masked_scene_attrs = (masks * scene_attrs).flatten(1).sum(dim=1)
            result[m_name]["scene_attrs"].append(masked_scene_attrs[correctly_classified].cpu().detach().numpy())
            result[m_name]["mask_size"].append(mask_size[correctly_classified].cpu().detach().numpy())
    return {m_name: {"object_attrs": np.concatenate(result[m_name]["object_attrs"]),
                     "scene_attrs": np.concatenate(result[m_name]["scene_attrs"]),
                     "mask_size": np.concatenate(result[m_name]["mask_size"])}
            for m_name in methods}