from typing import Callable
import torch


def input_dependence_rate(overlays: torch.Tensor, scenes: torch.Tensor, masks: torch.Tensor, scene_labels: torch.Tensor,
                          model: Callable, method: Callable, device: str):
    """
    Input dependence rate:
    Percentage of correctly classified inputs with object overlay by model trained on scene labels,
    where region of object overlay is attributed less than that same region when the overlay
    is removed. 1 - IDR can be interpreted as "false positive rate": rate of explanations
    that assign higher importance to less important features.
    """
    overlays = overlays.to(device)
    scenes = scenes.to(device)
    labels = scene_labels.to(device)
    masks = masks.squeeze().to(device)
    with torch.no_grad():
        y_pred = torch.argmax(model(overlays), dim=1)
    correctly_classified = (y_pred == labels)

    object_attrs = method(overlays, labels)
    scene_attrs = method(scenes, labels)
    masked_object_attrs = (masks * object_attrs).flatten(1).sum(dim=1)
    masked_scene_attrs = (masks * scene_attrs).flatten(1).sum(dim=1)
    correct = correctly_classified & (masked_scene_attrs > masked_object_attrs)

    return correct, correctly_classified
