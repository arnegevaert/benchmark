from torch import multiprocessing as mp
from torch import distributed as dist


class Worker:
    def __init__(self, result_queue: mp.Queue, rank: int, world_size: int, all_processes_done: mp.Event):
        self.result_queue = result_queue
        self.rank = rank
        self.world_size = world_size
        self.all_processes_done = all_processes_done

    def setup(self):
        dist.init_process_group("gloo", rank=self.rank, world_size=self.world_size)

    def cleanup(self):
        self.all_processes_done.wait()
        dist.destroy_process_group()

    def run(self):
        self.setup()
        self.work()
        self.cleanup()

    def work(self):
        raise NotImplementedError