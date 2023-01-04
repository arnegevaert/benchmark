from torch.utils.data import DataLoader
import torch
from attrbench.distributed.distributed_sampler import DistributedSampler
from attrbench.distributed.metrics.metric_worker import MetricWorker
from torch import multiprocessing as mp
from typing import Callable
from attrbench.data import HDF5Dataset
from torch import nn


class ImpactCoverageWorker(MetricWorker):
    def __init__(self, result_queue: mp.Queue, rank: int, world_size: int, 
                 all_processes_done, model_factory: Callable[[], nn.Module],
                 dataset: HDF5Dataset, batch_size: int,
                 patch_folder: str):
        super().__init__(result_queue, rank, world_size, all_processes_done, 
                         model_factory, dataset, batch_size)

    def work(self):
        model = self._get_model()
        sampler = DistributedSampler(self.dataset, self.world_size, self.rank)
        dataloader = DataLoader(self.dataset, sampler=sampler,
                                batch_size=self.batch_size, num_workers=4,
                                pin_memory=True)
        device = torch.device(self.rank)

        # TODO Load the patches from the patch folder
        pass

        for batch_indices, batch_x, batch_y in self.dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_result = {}
