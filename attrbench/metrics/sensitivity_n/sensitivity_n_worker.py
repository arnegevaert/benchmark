from torch import multiprocessing as mp
import numpy as np
import torch
from typing import Callable, Dict, Tuple
from torch import nn
from attrbench.data import AttributionsDataset
from attrbench.masking import Masker
from attrbench.metrics import MetricWorker
from attrbench.metrics.result import BatchResult
from attrbench.distributed import PartialResultMessage, DoneMessage
from attrbench.metrics.sensitivity_n._dataset import _SensitivityNDataset, _SegSensNDataset
from attrbench.util.segmentation import segment_attributions
from attrbench.util.util import ACTIVATION_FNS
from attrbench.stat import corrcoef


class SensitivityNWorker(MetricWorker):
    def __init__(self, result_queue: mp.Queue, rank: int, world_size: int, all_processes_done: mp.Event,
                 model_factory: Callable[[], nn.Module], dataset: AttributionsDataset, batch_size: int,
                 min_subset_size: float, max_subset_size: float, num_steps: int, num_subsets: int,
                 maskers: Dict[str, Masker], activation_fns: Tuple[str],
                 segmented=False):
        super().__init__(result_queue, rank, world_size, all_processes_done, model_factory, dataset, batch_size)
        self.activation_fns = activation_fns
        self.maskers = maskers
        self.num_subsets = num_subsets
        self.num_steps = num_steps
        self.max_subset_size = max_subset_size
        self.min_subset_size = min_subset_size
        self.segmented = segmented

    def work(self):
        model = self._get_model()

        for batch_indices, batch_x, batch_y, batch_attr in self.dataloader:
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
                        activation_fn: [] for activation_fn in self.activation_fns
                    } for masker in self.maskers.keys()
                } for method_name in self.dataset.method_names
            }

            # Compute range of numbers of features to remove
            n_range = np.linspace(self.min_subset_size, self.max_subset_size, self.num_steps)
            if self.segmented:
                n_range = n_range * 100
            else:
                total_num_features = np.prod(self.dataset.attributions_shape)
                n_range = n_range * total_num_features
            n_range = n_range.astype(np.int)

            for masker_name, masker in self.maskers.items():
                """
                batch_result[masker_name] = sensitivity_n(batch_x, batch_y, model, batch_attr.numpy(),
                                                          self.min_subset_size, self.max_subset_size, self.num_steps,
                                                          self.num_subsets, masker, self.activation_fns)
                """
                # Create pseudo-dataset to generate perturbed samples
                if self.segmented:
                    ds = _SegSensNDataset(n_range, self.num_subsets, batch_x)
                    ds.set_masker(masker)
                else:
                    ds = _SensitivityNDataset(n_range, self.num_subsets, batch_x, masker)

                # Calculate differences in output and removed indices (will be re-used for all methods)
                # activation_fn -> n -> [batch_size, 1]
                output_diffs = {activation_fn: {n: [] for n in n_range} for activation_fn in self.activation_fns}
                removed_indices = {n: [] for n in n_range}
                for i in range(len(ds)):
                    batch, indices, n = ds[i]
                    n = n.item()
                    with torch.no_grad():
                        output = model(batch)
                    for activation_fn in self.activation_fns:
                        activated_output = ACTIVATION_FNS[activation_fn](output)
                        # [batch_size, 1]
                        output_diffs[activation_fn][n].append(
                            (activated_orig_output[activation_fn] - activated_output)
                            .gather(dim=1, index=batch_y.unsqueeze(-1))
                        )
                    removed_indices[n].append(indices)  # [batch_size, n]
                # All output differences have been computed. Concatenate/stack all results into numpy arrays
                for n in n_range:
                    for activation_fn in self.activation_fns:
                        # [batch_size, num_subsets]
                        output_diffs[activation_fn][n] = torch.cat(output_diffs[activation_fn][n], dim=1)\
                            .detach().cpu().numpy()
                    # [batch_size, num_subsets, n]
                    removed_indices[n] = np.stack(removed_indices[n], axis=1)

                # Compute correlations for all methods
                # TODO this might have to happen in another process, not sure if possible?
                for method_name in self.dataset.method_names:
                    attrs = batch_attr[method_name].detach().cpu().numpy()
                    if self.segmented:
                        attrs = segment_attributions(ds.segmented_images.cpu().numpy(), attrs)
                    # [batch_size, 1, -1]
                    attrs = attrs.reshape((attrs.shape[0], 1, -1))
                    for n in n_range:
                        # [batch_size, num_subsets, n]
                        n_mask_attrs = np.take_along_axis(attrs, axis=-1, indices=removed_indices[n])
                        for activation_fn in self.activation_fns:
                            # Compute sum of attributions
                            n_sum_of_attrs = n_mask_attrs.sum(axis=-1)  # [batch_size, num_subsets]
                            n_output_diffs = output_diffs[activation_fn][n]
                            # Compute correlation between output difference and sum of attribution values
                            batch_result[method_name][masker_name][activation_fn].append(
                                corrcoef(n_sum_of_attrs, n_output_diffs))
                    for activation_fn in self.activation_fns:
                        afn_result = batch_result[method_name][masker_name][activation_fn]
                        # [batch_size, len(n_range)]
                        stacked_result = torch.tensor(np.stack(afn_result, axis=1))
                        batch_result[method_name][masker_name][activation_fn] = stacked_result

            self.result_queue.put(
                PartialResultMessage(self.rank, BatchResult(batch_indices, batch_result))
            )
        self.result_queue.put(DoneMessage(self.rank))
