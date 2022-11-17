from attrbench.distributed import Worker, DistributedSampler, DoneMessage, DistributedComputation, PartialResultMessage
from torch import multiprocessing as mp


class SensitivityNWorker(Worker):
    def __init__(self):
        pass

    def work(self):
        pass


class DistributedSensitivityN(DistributedComputation):
    def __init__(self):
        pass

    def run(self):
        pass

    def save_result(self, path: str):
        pass

    def _create_worker(self, queue: mp.Queue, rank: int, all_processes_done: mp.Event) -> Worker:
        pass

    def _handle_result(self, result: PartialResultMessage):
        pass