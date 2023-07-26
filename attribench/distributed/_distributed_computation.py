from abc import abstractmethod
from ._worker import (
    Worker,
    WorkerConfig,
    DistributedWorkerConfig,
    SynchronizedWorkerConfig,
)
from ._message import PartialResultMessage, DoneMessage
import torch.multiprocessing as mp
from typing import Optional, Tuple
import torch
import os


class DistributedComputation:
    def __init__(
        self,
        address: str,
        port: str | int,
        devices: Optional[Tuple[int]] = None,
    ):
        self.address = address
        self.port = str(port)
        self.devices = (
            devices
            if devices is not None
            else list(range(torch.cuda.device_count()))
        )
        self.world_size = len(self.devices)
        self.ctx = mp.get_context("spawn")

    @abstractmethod
    def _handle_result(self, result: PartialResultMessage):
        raise NotImplementedError

    @abstractmethod
    def _create_worker(self, worker_config: WorkerConfig) -> Worker:
        raise NotImplementedError

    def _cleanup(self):
        pass

    def run(self, *args, **kwargs):
        if self.world_size != 1:
            # Initialize multiproc parameters
            os.environ["MASTER_ADDR"] = self.address
            os.environ["MASTER_PORT"] = self.port

            result_queue = self.ctx.Queue()  # Will store results from metric
            # Used for signaling processes that they can terminate
            all_processes_done = self.ctx.Event()
            processes = []  # Used for joining all processes

            # Start all processes
            for rank in range(self.world_size):
                worker_config = DistributedWorkerConfig(
                    self.world_size, result_queue, rank, all_processes_done
                )
                worker = self._create_worker(worker_config)
                p = self.ctx.Process(target=worker.run)
                p.start()
                processes.append(p)

            # Gather results
            # This is not busy waiting:
            # queue.get will block until a result is passed
            done_processes = [False for _ in processes]
            while not all(done_processes):
                # res is a PartialResultMessage or a DoneMessage
                res = result_queue.get()
                if isinstance(res, DoneMessage):
                    done_processes[res.rank] = True
                else:
                    self._handle_result(res)

            # Processes are now allowed to terminate
            all_processes_done.set()
            for p in processes:
                p.join()
            self._cleanup()
        else:
            # If world size is 1, run on the main process
            worker_config = SynchronizedWorkerConfig(self._handle_result)
            worker = self._create_worker(worker_config)
            worker.run()
            self._cleanup()
