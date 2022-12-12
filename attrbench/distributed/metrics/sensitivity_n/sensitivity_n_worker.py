from torch import multiprocessing as mp
import numpy as np
import torch
from typing import Callable, Dict, Tuple
from torch import nn
from attrbench.data import AttributionsDataset
from attrbench.masking import Masker
from attrbench.distributed.metrics import MetricWorker
from attrbench.distributed.metrics.result import BatchResult
from attrbench.distributed import PartialResultMessage, DoneMessage
from attrbench.metrics.sensitivity_n._dataset import _SensitivityNDataset
from attrbench.util.util import ACTIVATION_FNS


class SensitivityNWorker(MetricWorker):
    def __init__(self, result_queue: mp.Queue, rank: int, world_size: int, all_processes_done: mp.Event,
                 model_factory: Callable[[], nn.Module], dataset: AttributionsDataset, batch_size: int,
                 min_subset_size: float, max_subset_size: float, num_steps: int, num_subsets: int,
                 maskers: Dict[str, Masker], activation_fns: Tuple[str]):
        super().__init__(result_queue, rank, world_size, all_processes_done, model_factory, dataset, batch_size)
        self.activation_fns = activation_fns
        self.maskers = maskers
        self.num_subsets = num_subsets
        self.num_steps = num_steps
        self.max_subset_size = max_subset_size
        self.min_subset_size = min_subset_size

    def work(self):
        model = self._get_model()

        for batch_indices, batch_x, batch_y, batch_attr, method_names in self.dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            activated_orig_output = {}
            with torch.no_grad():
                orig_output = model(batch_x)
                for activation_fn in self.activation_fns:
                    activated_orig_output[activation_fn] = ACTIVATION_FNS[activation_fn](orig_output)

            # method_name -> masker -> activation_fn -> [batch_size, num_steps]
            batch_result: Dict[str, Dict[str, torch.Tensor]] = {
                method_name: {
                    masker: {
                        activation_fn: None for activation_fn in self.activation_fns
                    } for masker in self.maskers.keys()
                } for method_name in self.dataset.method_names
            }

            # Compute range of numbers of features to remove
            total_num_features = np.prod(self.dataset.attributions_shape)
            n_range = np.linspace(self.min_subset_size, self.max_subset_size, self.num_steps)
            n_range = (n_range * total_num_features).astype(np.int)

            for masker_name, masker in self.maskers.items():
                """
                batch_result[masker_name] = sensitivity_n(batch_x, batch_y, model, batch_attr.numpy(),
                                                          self.min_subset_size, self.max_subset_size, self.num_steps,
                                                          self.num_subsets, masker, self.activation_fns)
                """
                # Create pseudo-dataset to generate perturbed samples
                ds = _SensitivityNDataset(n_range, self.num_subsets, batch_x, masker)

                # Calculate differences in output and removed indices (will be re-used for all methods)
                output_diffs = {activation_fn: {n: [] for n in n_range} for activation_fn in self.activation_fns}
                removed_indices = {n: [] for n in n_range}
                for i in range(len(ds)):
                    batch, indices, n = ds[i]
                    n = n.item()
                    with torch.no_grad():
                        output = model(batch)
                    for activation_fn in self.activation_fns:
                        activated_output = ACTIVATION_FNS[activation_fn](output)

            self.result_queue.put(
                PartialResultMessage(self.rank, BatchResult(batch_indices, batch_result, method_names))
            )
        self.result_queue.put(DoneMessage(self.rank))