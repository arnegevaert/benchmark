from torch.utils.data import DataLoader
from attrbench.data import IndexDataset
import numpy as np
import logging
import random
import re
from itertools import cycle
import os
import torch
from attrbench.distributed.distributed_sampler import DistributedSampler
from attrbench.distributed.message import DoneMessage, PartialResultMessage
from attrbench.distributed.metrics.metric_worker import MetricWorker
from torch import multiprocessing as mp
from typing import Callable, Dict, NewType
from torch import nn

from attrbench.distributed.metrics.result.batch_result import BatchResult


AttributionMethod = NewType("AttributionMethod", Callable[[torch.Tensor, torch.Tensor], torch.Tensor])


class ImpactCoverageWorker(MetricWorker):
    def __init__(self, result_queue: mp.Queue, rank: int, world_size: int, 
                 all_processes_done, model_factory: Callable[[], nn.Module],
                 method_factory: Callable[[nn.Module], Dict[str, AttributionMethod]],
                 dataset: IndexDataset, batch_size: int,
                 patch_folder: str):
        super().__init__(result_queue, rank, world_size, all_processes_done, 
                         model_factory, dataset, batch_size)
        self.patch_folder = patch_folder
        self.method_factory = method_factory

    def work(self):
        model = self._get_model()
        sampler = DistributedSampler(self.dataset, self.world_size, self.rank)
        dataloader = DataLoader(self.dataset, sampler=sampler,
                                batch_size=self.batch_size, num_workers=4,
                                pin_memory=True)
        device = torch.device(self.rank)

        # Get method dictionary
        method_dict = self.method_factory(model)

        # Get names of patches and compile regular expression for deriving target labels
        patch_names = cycle([filename for filename in os.listdir(self.patch_folder) 
                             if filename.endswith(".pt")])
        target_expr = re.compile(r".*_([0-9]*)\.pt")

        for batch_indices, batch_x, batch_y in dataloader:
            batch_result: Dict[str, torch.Tensor] = {
                        method_name: None for method_name in method_dict.keys()
                    }
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Get original output and initialize datastructures
            with torch.no_grad():
                original_output = model(batch_x).detach().cpu()
            successful = torch.zeros(batch_x.shape[0]).bool()
            attacked_samples = batch_x.clone()
            targets = torch.zeros(batch_y.shape).long()
            patch_mask = torch.zeros(batch_x.shape)
            max_tries = 50
            num_tries = 0

            # Apply patches to images
            while not torch.all(successful):
                num_tries +=1
                # Load next patch
                patch_name = next(patch_names)
                target = int(target_expr.match(patch_name).group(1))
                patch = torch.load(
                        os.path.join(self.patch_folder, patch_name),
                        map_location=lambda storage, _: storage
                        ).to(device)
                image_size = batch_x.shape[-1]
                patch_size = patch.shape[-1]

                # Apply patch to all images in batch (random location, but same for each image in batch)
                indx = random.randint(0, image_size - patch_size)
                indy = random.randint(0, image_size - patch_size)
                attacked_samples[~successful, ...] = batch_x[~successful, ...].clone()
                attacked_samples[~successful, :, indx:indx + patch_size, indy:indy + patch_size] = patch.float()
                with torch.no_grad():
                    adv_out = model(attacked_samples).detach().cpu()
                
                # Set the patch mask and targets for the samples that were successful this iteration
                # We set the patch mask for all samples that weren't yet successful
                # This way, if any samples can't be attacked, they will still have a patch on them
                # (even though it didn't flip the prediction)
                patch_mask[~successful, ...] = 0
                patch_mask[~successful, :, indx:indx + patch_size, indy:indy + patch_size] = 1
                targets[~successful] = target

                # Add the currently successful samples to all successful samples
                successful_now = (original_output.argmax(axis=1) != target) &\
                                 (adv_out.argmax(axis=1) == target) &\
                                 (batch_y.cpu() != target)
                successful = successful | successful_now

                if num_tries > max_tries:
                    logging.warning(
                        f"Not all samples could be attacked: {torch.sum(successful)}/{batch_x.size(0)} were successful.")
                    break
            targets = targets.to(device)

            # Compute impact coverage for each method
            for method_name, method in method_dict.items():
                attrs = method(attacked_samples, target=targets).detach().cpu().numpy()

                # Check attributions shape
                if attrs.shape[1] not in (1, 3):
                    raise ValueError(f"Impact Coverage only works on image data. Attributions must have 1 or 3 color channels."
                                     f"Found attributions shape {attrs.shape}.")
                # If attributions have only 1 color channel, we need a single-channel patch mask as well
                if attrs.shape[1] == 1:
                    patch_mask = patch_mask[:, 0, :, :]

                # Get indices of top k attributions
                flattened_attrs = attrs.reshape(attrs.shape[0], -1)
                sorted_indices = flattened_attrs.argsort()
                # Number of top attributions is equal to number of features masked by the patch
                # We assume here that the mask is the same size for all samples!
                nr_top_attributions = patch_mask[0, ...].long().sum().item()

                # Create mask of critical factors (most important pixels/features according to attributions)
                to_mask = sorted_indices[:, -nr_top_attributions:]
                critical_factor_mask = np.zeros(attrs.shape).reshape(attrs.shape[0], -1)
                batch_size = attrs.shape[0]
                batch_dim = np.tile(range(batch_size), (nr_top_attributions, 1)).transpose()
                critical_factor_mask[batch_dim, to_mask] = 1
                critical_factor_mask = critical_factor_mask.astype(np.bool)

                # Calculate IoU of critical factors (top n attributions) with adversarial patch
                patch_mask_flattened = patch_mask.flatten(1).bool().numpy()
                intersection = (patch_mask_flattened & critical_factor_mask).sum(axis=1)
                union = (patch_mask_flattened | critical_factor_mask).sum(axis=1)
                iou = intersection.astype(np.float) / union.astype(np.float)
                batch_result[method_name] = iou

            # Return batch result
            self.result_queue.put(
                PartialResultMessage(self.rank, BatchResult(batch_indices, batch_result))
            )
        self.result_queue.put(DoneMessage(self.rank))

