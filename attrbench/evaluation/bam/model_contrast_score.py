from typing import Iterable, Callable, Dict
from tqdm import tqdm
import torch
import numpy as np
from attrbench.evaluation.result import BoxPlotResult, NumpyJSONEncoder
import json
import matplotlib.pyplot as plt


def model_contrast_score(dataloader: Iterable, object_model: Callable,
                         object_methods: Dict[str, Callable],
                         scene_methods: Dict[str, Callable], device: str):
    """
    Model contrast score:
    Difference of importance of object pixels for model trained on object labels
    (should be important) and model trained on scene labels (should not be important)
    """
    object_attrs = {m_name: [] for m_name in object_methods}
    scene_attrs = {m_name: [] for m_name in object_methods}
    mask_sizes = {m_name: [] for m_name in object_methods}
    for images, masks, _, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.squeeze().to(device)
        with torch.no_grad():
            y_pred = torch.argmax(object_model(images), dim=1)
        # Boolean array indicating images correctly classified by object model
        correctly_classified = (y_pred == labels)
        for m_name in object_methods:
            object_model_attrs = object_methods[m_name](images, labels)
            scene_model_attrs = scene_methods[m_name](images, labels)
            mask_size = torch.sum(masks.flatten(1), dim=1)
            object_attrs[m_name].append((masks * object_model_attrs)[correctly_classified].flatten(1).sum(dim=1)
                                                  .cpu().detach().numpy())
            scene_attrs[m_name].append((masks * scene_model_attrs)[correctly_classified].flatten(1).sum(dim=1)
                                                 .cpu().detach().numpy())
            mask_sizes[m_name].append(mask_size[correctly_classified].cpu().detach().numpy())
    return MCSResult(np.concatenate(object_attrs),
                     np.concatenate(scene_attrs),
                     np.concatenate(mask_sizes))


class MCSResult(BoxPlotResult):
    def __init__(self, object_attrs, scene_attrs, mask_sizes):
        self.object_attrs = object_attrs
        self.scene_attrs = scene_attrs
        self.mask_sizes = mask_sizes
        super().__init__({
            method: (object_attrs[method] - scene_attrs[method]) / mask_sizes[method]
            for method in object_attrs
        })

    def save_json(self, filename):
        with open(filename, "w") as outfile:
            json.dump({
                "object_attrs": self.object_attrs,
                "scene_attrs": self.scene_attrs,
                "mask_sizes": self.mask_sizes
            }, outfile, cls=NumpyJSONEncoder)
