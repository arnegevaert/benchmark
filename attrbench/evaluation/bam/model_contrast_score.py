from typing import Iterable, Callable, Dict
from tqdm import tqdm
import torch
import numpy as np


# TODO only save samples that object model classifies correctly?
def model_contrast_score(dataloader: Iterable, object_methods: Dict[str, Callable],
                         scene_methods: Dict[str, Callable], device: str):
    """
    Model contrast score:
    Difference of importance of object pixels for model trained on object labels
    (should be important) and model trained on scene labels (should not be important)
    """
    result = {m_name: {"object_attrs": [], "scene_attrs": [], "mask_size": []}
              for m_name in object_methods}
    for images, masks, _, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.squeeze().to(device)
        for m_name in object_methods:
            object_model_attrs = object_methods[m_name](images, labels)
            scene_model_attrs = scene_methods[m_name](images, labels)
            mask_size = torch.sum(masks.flatten(1), dim=1)
            result[m_name]["object_attrs"].append((masks * object_model_attrs).flatten(1).sum(dim=1)
                                                  .cpu().detach().numpy())
            result[m_name]["scene_attrs"].append((masks * scene_model_attrs).flatten(1).sum(dim=1)
                                                 .cpu().detach().numpy())
            result[m_name]["mask_size"].append(mask_size.cpu().detach().numpy())
    return {m_name: {"object_attrs": np.concatenate(result[m_name]["object_attrs"]),
                     "scene_attrs": np.concatenate(result[m_name]["scene_attrs"]),
                     "mask_size": np.concatenate(result[m_name]["mask_size"])}
            for m_name in object_methods}