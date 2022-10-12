from attrbench.distributed import PartialResultMessage, DoneMessage
import torch.distributed as dist
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

    def _worker(self, queue: mp.Queue, rank: int, **kwargs):
        raise NotImplementedError

    def _handle_result(self, result: PartialResultMessage):
        raise NotImplementedError

    def _worker_setup(self, queue: mp.Queue, rank: int, done_event: mp.Event):
        dist.init_process_group("gloo", rank=rank, world_size=self.world_size)
        self._worker(queue, rank)
        done_event.wait()
        dist.destroy_process_group()  # TODO if we destroy the process group in the start() method, is the event still necessary?

    def start(self):
        # Initialize multiproc parameters
        os.environ["MASTER_ADDR"] = self.address
        os.environ["MASTER_PORT"] = self.port
        mp.set_start_method("spawn")

        queue = mp.Queue()  # Will store results from metric
        global_done_event = mp.Event()  # Used for signaling processes that they can terminate
        processes = []  # Used for joining all processes

        # Start all processes
        for rank in range(self.world_size):
            p = mp.Process(target=self._worker_setup, args=(queue, rank, global_done_event))
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
        global_done_event.set()
        for p in processes:
            p.join()