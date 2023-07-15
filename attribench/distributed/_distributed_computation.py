from ._worker import Worker
from ._message import PartialResultMessage, DoneMessage
import torch.multiprocessing as mp
from typing import Optional, Tuple
import torch
import os
from multiprocessing.synchronize import Event


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

    def _handle_result(self, result: PartialResultMessage):
        raise NotImplementedError

    def _create_worker(
        self, queue: mp.Queue, rank: int, all_processes_done: Event
    ) -> Worker:
        raise NotImplementedError

    def _cleanup(self):
        pass

    def run(self):
        if self.world_size != 1:
            # Initialize multiproc parameters
            os.environ["MASTER_ADDR"] = self.address
            os.environ["MASTER_PORT"] = self.port
            ctx = mp.get_context("spawn")

            queue = ctx.Queue()  # Will store results from metric
            # Used for signaling processes that they can terminate
            all_processes_done = ctx.Event()
            processes = []  # Used for joining all processes

            # Start all processes
            for rank in range(self.world_size):
                worker = self._create_worker(queue, rank, all_processes_done)
                p = ctx.Process(target=worker.run)
                p.start()
                processes.append(p)

            # Gather results
            # This is not busy waiting:
            # queue.get will block until a result is passed
            done_processes = [False for _ in processes]
            while not all(done_processes):
                # res is a PartialResultMessage or a DoneMessage
                res = queue.get()
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
            worker = self._create_worker(None, 0, None)
            worker.run()
            self._cleanup()
