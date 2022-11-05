from attrbench.distributed import PartialResultMessage, DoneMessage, Worker
import torch.multiprocessing as mp
from typing import Tuple
import torch
import os


class DistributedComputation:
    def __init__(self, address="localhost", port="12355", devices: Tuple[int] = None):
        self.address = address
        self.port = port
        self.devices = devices if devices is not None else list(range(torch.cuda.device_count()))
        self.world_size = len(self.devices)
        self.ctx = mp.get_context("spawn")

    def _handle_result(self, result: PartialResultMessage):
        raise NotImplementedError

    def _create_worker(self, queue: mp.Queue, rank: int, all_processes_done: mp.Event) -> Worker:
        raise NotImplementedError

    def run(self):
        # Initialize multiproc parameters
        os.environ["MASTER_ADDR"] = self.address
        os.environ["MASTER_PORT"] = self.port
        ctx = mp.get_context("spawn")

        queue = ctx.Queue()  # Will store results from metric
        all_processes_done = ctx.Event()  # Used for signaling processes that they can terminate
        processes = []  # Used for joining all processes

        # Start all processes
        for rank in range(self.world_size):
            worker = self._create_worker(queue, rank, all_processes_done)
            p = ctx.Process(target=worker.run)
            p.start()
            processes.append(p)

        # Gather results
        # This is not busy waiting: queue.get will block until a result is passed
        done_processes = [False for _ in processes]
        while not all(done_processes):
            res = queue.get()  # res is a PartialResultMessage or a DoneMessage
            if type(res) == DoneMessage:
                done_processes[res.rank] = True
            else:
                self._handle_result(res)

        # Processes are now allowed to terminate
        all_processes_done.set()
        for p in processes:
            p.join()
