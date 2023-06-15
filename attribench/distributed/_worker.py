from torch import multiprocessing as mp
from torch import distributed as dist
from ._message import PartialResultMessage, DoneMessage
from typing import Optional, Callable
from multiprocessing.synchronize import Event


class Worker:
    def __init__(
        self,
        result_queue: mp.Queue,
        rank: int,
        world_size: int,
        all_processes_done: Event,
        result_handler: Optional[
            Callable[[PartialResultMessage], None]
        ] = None,
    ):
        """
        :param result_queue: The queue to put results in. Is read by the
            DistributedComputation in the main process.
        :param rank: The process number of this worker.
        :param world_size: The total number of processes.
        :param all_processes_done: An event that is set by the
            DistributedComputation when all processes are done.
            Signals that the worker can terminate.
        :param synchronized: If True, no multiprocessing is used and the worker
            is run on the main process. This means that _handle_result is
            called directly instead of putting the result in the queue.
        """
        self.result_queue = result_queue
        self.rank = rank
        self.world_size = world_size
        self.all_processes_done = all_processes_done
        self.synchronized = world_size == 1
        if self.synchronized and result_handler is None:
            raise ValueError(
                "If world_size is 1, result_handler must be provided."
            )
        elif result_handler is not None:
            self.result_handler = result_handler

    def setup(self):
        if not self.synchronized:
            dist.init_process_group(
                "gloo", rank=self.rank, world_size=self.world_size
            )

    def cleanup(self):
        if not self.synchronized:
            self.result_queue.put(DoneMessage(self.rank))
            self.all_processes_done.wait()
            dist.destroy_process_group()

    def send_result(self, result: PartialResultMessage):
        if self.synchronized:
            self.result_handler(result)
        else:
            self.result_queue.put(result)

    def run(self):
        self.setup()
        self.work()
        self.cleanup()

    def work(self):
        raise NotImplementedError
