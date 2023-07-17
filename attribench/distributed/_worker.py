from abc import abstractmethod
from torch import multiprocessing as mp
from torch import distributed as dist
from ._message import PartialResultMessage, DoneMessage
from typing import Callable
from multiprocessing.synchronize import Event


class WorkerConfig:
    def __init__(self, world_size: int, rank: int) -> None:
        self.world_size = world_size
        self.rank = rank

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def cleanup(self):
        raise NotImplementedError

    @abstractmethod
    def send_result(self, result: PartialResultMessage):
        raise NotImplementedError


class DistributedWorkerConfig(WorkerConfig):
    def __init__(
        self,
        world_size: int,
        result_queue: mp.Queue,
        rank: int,
        all_processes_done: Event,
    ):
        super().__init__(world_size, rank)
        self.result_queue = result_queue
        self.all_processes_done = all_processes_done

    def setup(self):
        dist.init_process_group(
            "gloo", rank=self.rank, world_size=self.world_size
        )

    def cleanup(self):
        self.result_queue.put(DoneMessage(self.rank))
        self.all_processes_done.wait()
        dist.destroy_process_group()

    def send_result(self, result: PartialResultMessage):
        self.result_queue.put(result)


class SynchronizedWorkerConfig(WorkerConfig):
    def __init__(self, result_handler: Callable[[PartialResultMessage], None]):
        super().__init__(1, 0)
        self.result_handler = result_handler

    def setup(self):
        pass

    def cleanup(self):
        pass

    def send_result(self, result: PartialResultMessage):
        self.result_handler(result)


class Worker:
    def __init__(
        self,
        worker_config: WorkerConfig,
    ):
        self.worker_config = worker_config

    def run(self):
        self.worker_config.setup()
        self.work()
        self.worker_config.cleanup()

    def work(self):
        raise NotImplementedError
