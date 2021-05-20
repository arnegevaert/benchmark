from typing import Callable, List, Union, Tuple, Dict

import numpy as np
import torch

from attrbench.lib import AttributionWriter
from attrbench.lib.masking import Masker
from attrbench.metrics import MaskerMetric
from ._dataset import _DeletionDataset
from .result import DeletionResult
from attrbench.lib.util import ACTIVATION_FNS


def deletion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
             num_steps: int, masker: Masker,
             activation_fns: Union[Tuple[str], str] = "linear",
             writer: AttributionWriter = None) -> Dict:
    if type(activation_fns) == str:
        activation_fns = (activation_fns,)
    ds = _DeletionDataset(num_steps, samples, attrs, masker)

    preds = {fn: [] for fn in activation_fns}
    for i in range(len(ds)):
        batch = ds[i]
        with torch.no_grad():
            predictions = model(batch)
        if writer is not None:
            writer.add_images("masked_samples", batch, global_step=i)
        for fn in activation_fns:
            preds[fn].append(ACTIVATION_FNS[fn](predictions).gather(dim=1, index=labels.unsqueeze(-1)).cpu())

    for afn in activation_fns:
        preds[afn] = torch.cat(preds[afn], dim=1).cpu()  # [batch_size, len(mask_range)]

    return preds


class _InsertionDeletion(MaskerMetric):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, maskers: Dict,
                 activation_fns: Union[Tuple[str], str],
                 writer_dir: str = None):
        super().__init__(model, method_names, maskers, writer_dir)
        self.num_steps = num_steps
        self.activation_fns = (activation_fns,) if type(activation_fns) == str else activation_fns
        self._result: DeletionResult = DeletionResult(method_names + ["_BASELINE"], list(self.maskers.keys()),
                                                      self.activation_fns)

    def run_batch(self, samples, labels, attrs_dict: dict, baseline_attrs: np.ndarray):
        methods_result = {masker_name: {afn: {} for afn in self.activation_fns} for masker_name in self.maskers}
        baseline_result = {masker_name: {afn: [] for afn in self.activation_fns} for masker_name in self.maskers}
        for masker_name, masker in self.maskers.items():
            methods_result[masker_name] = {afn: {} for afn in self.activation_fns}
            for method_name in attrs_dict:
                result = deletion(samples, labels, self.model,
                                  attrs_dict[method_name], self.num_steps, masker,
                                  self.activation_fns,
                                  self._get_writer(method_name))
                for afn in self.activation_fns:
                    methods_result[masker_name][afn][method_name] = result[afn].cpu().detach().numpy()

            for i in range(baseline_attrs.shape[0]):
                bl_result = deletion(samples, labels, self.model,
                                     baseline_attrs[i, ...], self.num_steps, masker,
                                     self.activation_fns)
                for afn in self.activation_fns:
                    baseline_result[masker_name][afn].append(bl_result[afn].cpu().detach().numpy())
            for afn in self.activation_fns:
                baseline_result[masker_name][afn] = np.stack(baseline_result[masker_name][afn], axis=1)
        self.result.append(methods_result)
        self.result.append(baseline_result, method="_BASELINE")


class Deletion(_InsertionDeletion):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, maskers: Dict,
                 activation_fns: Union[Tuple[str], str], writer_dir: str = None):
        super().__init__(model, method_names, num_steps, maskers, activation_fns,
                         writer_dir)
