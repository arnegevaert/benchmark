from typing import Callable, Dict, List
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from attrbench.evaluation.bam import BAMDataset
from attrbench.evaluation.result import BoxPlotResult
import json
from itertools import tee


def input_dependence_rate(dataset: BAMDataset, batch_size: int, models: List[torch.nn.Module], methods: List[Dict[str, Callable]],
                          device: str):
    """
    Input dependence rate:
    Percentage of correctly classified inputs with object overlay by model trained on scene labels,
    where region of object overlay is attributed less than that same region when the overlay
    is removed. 1 - IDR can be interpreted as "false positive rate": rate of explanations
    that assign higher importance to less important features.
    """
    dataset.include_orig_scene = True
    dataset.include_mask = True
    m_names = methods[0].keys()
    result = {m_name: [] for m_name in m_names}
    for model_index, cur_model in enumerate(models):
        print(f"Model {model_index+1}/{len(models)}:")
        models[model_index].to(device)
        cur_methods = methods[model_index]
        cur_result = {m_name: {"correct": 0, "total": 0} for m_name in m_names}
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
        for images, scenes, masks, scene_labels, object_labels in tqdm(dataloader):
            images = images.to(device)
            scenes = scenes.to(device)
            labels = scene_labels.to(device)
            masks = masks.squeeze().to(device)  # [batch_size, 1, height, width]
            with torch.no_grad():
                y_pred = torch.argmax(cur_model(images), dim=1)
            # Boolean array indicating correctly classified images
            correctly_classified = (y_pred == labels)
            for m_name in cur_methods:
                object_attrs = cur_methods[m_name](images, labels)  # [batch_size, *sample_shape]
                scene_attrs = cur_methods[m_name](scenes, labels)  # [batch_size, *sample_shape]
                masked_object_attrs = (masks * object_attrs).flatten(1).sum(dim=1)
                masked_scene_attrs = (masks * scene_attrs).flatten(1).sum(dim=1)
                cur_result[m_name]["correct"] += (correctly_classified & (masked_scene_attrs > masked_object_attrs))\
                    .sum().item()
                cur_result[m_name]["total"] += correctly_classified.sum().item()
        models[model_index].to("cpu")
        for m_name in m_names:
            result[m_name].append(cur_result[m_name]["correct"] / cur_result[m_name]["total"])
    return BoxPlotResult(result)
