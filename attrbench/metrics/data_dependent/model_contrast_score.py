from typing import Callable
import torch


def model_contrast_score(overlays: torch.Tensor, masks: torch.Tensor, object_labels: torch.Tensor,
                         scene_labels: torch.Tensor, object_model: Callable, scene_model: Callable,
                         object_method: Callable, scene_method: Callable, device: str):
    """
    Model contrast score:
    Difference of importance of object pixels for model trained on object labels
    (should be important) and model trained on scene labels (should not be important)
    """
    overlays = overlays.to(device)
    object_labels = object_labels.to(device)
    scene_labels = scene_labels.to(device)
    masks = masks.squeeze().to(device)
    # We check if both the object model and the scene model make the correct classification
    with torch.no_grad():
        y_pred_obj = torch.argmax(object_model(overlays), dim=1)
        y_pred_scene = torch.argmax(scene_model(overlays), dim=1)
    correctly_classified = ((y_pred_obj == object_labels) & (y_pred_scene == scene_labels))

    object_model_attrs = object_method(overlays, object_labels)
    scene_model_attrs = scene_method(overlays, scene_labels)
    mask_sizes = torch.sum(masks.flatten(1), dim=1)
    diffs = (object_model_attrs - scene_model_attrs) / mask_sizes
    return diffs.cpu(), correctly_classified.cpu()
