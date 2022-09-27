import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(20, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


def run_queue(rank, size, queue: mp.Queue, event: mp.Event):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="gloo", rank=rank, world_size=size)

    device = torch.device(rank)
    model = Model()
    model.to(device)
    model.eval()
    with torch.no_grad():
        input_tensor = torch.randn(size=(4, 20), device=device)
        out = model(input_tensor).cpu()
    queue.put((rank, out))
    event.wait()
    print(rank, "done waiting")


if __name__ == "__main__":
    size = 1
    processes = []
    mp.set_start_method("spawn")
    queue = mp.Queue()
    event = mp.Event()

    for rank in range(size):
        p = mp.Process(target=run_queue, args=(rank, size, queue, event))
        p.start()
        processes.append(p)

    for _ in range(size):
        res = queue.get()
        print("Main process received:", res, flush=True)

    print("Setting event in main thread")
    event.set()
    print("Event set in main thread")

    for p in processes:
        p.join()
