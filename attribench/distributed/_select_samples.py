from ._distributed_sampler import DistributedSampler
from ._distributed_computation import DistributedComputation
from ._message import PartialResultMessage
from ._worker import Worker, WorkerConfig
from attribench import ModelFactory
from tqdm import tqdm
from attribench.data import HDF5DatasetWriter
from attribench.functional._select_samples import _select_samples_batch
from typing import Callable, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import torch
from numpy import typing as npt
from torch import nn
from multiprocessing.synchronize import Event


class SamplesResult:
    def __init__(self, samples: npt.NDArray, labels: npt.NDArray):
        self.samples = samples
        self.labels = labels


class SampleSelectionWorker(Worker):
    def __init__(
        self,
        worker_config: WorkerConfig,
        sufficient_samples: Event,
        batch_size: int,
        dataset: Dataset,
        model_factory: Callable[[], nn.Module],
    ):
        super().__init__(worker_config)
        self.sufficient_samples = sufficient_samples
        self.model_factory = model_factory
        self.dataset = dataset
        self.batch_size = batch_size

    def work(self):
        sampler = DistributedSampler(
            self.dataset,
            self.worker_config.world_size,
            self.worker_config.rank,
        )
        dataloader = DataLoader(
            self.dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )
        device = torch.device(self.worker_config.rank)
        model = self.model_factory()
        model.to(device)
        it = iter(dataloader)

        for batch_x, batch_y in it:
            correct_samples, correct_labels = _select_samples_batch(
                batch_x, batch_y, model, device
            )
            result = SamplesResult(
                correct_samples.cpu().numpy(), correct_labels.cpu().numpy()
            )
            self.worker_config.send_result(
                PartialResultMessage(self.worker_config.rank, result)
            )
            if self.sufficient_samples.is_set():
                break


class SelectSamples(DistributedComputation):
    """Select correctly classified samples from a dataset and write them
    to a HDF5 file. This is done in a distributed fashion, i.e. each
    subprocess selects a part of the samples and writes them to the
    HDF5 file. The number of processes is determined by the number of
    devices.

    If you want to select correctly classified samples and return them,
    rather than storing them to a HDF5 file, use
    :func:`attribench.functional.select_samples` instead.
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        dataset: Dataset,
        num_samples: int,
        batch_size: int,
        address: str = "localhost",
        port: str = "12355",
        devices: Optional[Tuple] = None,
    ):
        """
        Parameters
        ----------
        model_factory : ModelFactory
            ModelFactory instance or callable that returns a model.
            Used to instantiate a model for each subprocess.
        dataset : Dataset
            Torch Dataset containing the samples and labels.
        writer : HDF5DatasetWriter
            Writer to write the samples and labels to.
        num_samples : int
            Number of correctly classified samples to select.
        batch_size : int
            Batch size per subprocess to use for the dataloader.
        address : str, optional
            Address to use for the multiprocessing connection,
            by default "localhost"
        port : str, optional
            Port to use for the multiprocessing connection, by default "12355"
        devices : Tuple, optional
            Devices to use. If None, then all available devices are used.
            By default None.
        """
        super().__init__(address, port, devices)
        self.model_factory = model_factory
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.sufficient_samples = self.ctx.Event()
        self.count = 0
        self.prog: tqdm | None = None
        self.writer: HDF5DatasetWriter | None = None

    def _create_worker(
        self, worker_config: WorkerConfig
    ):
        return SampleSelectionWorker(
            worker_config,
            self.sufficient_samples,
            self.batch_size,
            self.dataset,
            self.model_factory,
        )

    def run(self, path: str):
        """Run the sample selection.

        Parameters
        ----------
        path : str
            Path to the HDF5 file to write the samples to.
        """
        self.writer = HDF5DatasetWriter(path, self.num_samples)
        self.prog = tqdm(total=self.num_samples)
        super().run()

    def _handle_result(self, result: PartialResultMessage[SamplesResult]):
        assert self.writer is not None

        # If too many samples, truncate
        samples = result.data.samples
        labels = result.data.labels
        if self.count + samples.shape[0] > self.num_samples:
            samples = samples[: self.num_samples - self.count]
            labels = labels[: self.num_samples - self.count]
            self.sufficient_samples.set()

        # Write to disk
        self.writer.write(samples, labels)

        # Update progress bar
        self.count += samples.shape[0]
        if self.prog is not None:
            self.prog.update(samples.shape[0])
