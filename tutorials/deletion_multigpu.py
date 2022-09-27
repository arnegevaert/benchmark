from util.get_dataset_model import get_dataloader, get_model
import torch
from attrbench.metrics import deletion
from attrbench.lib.masking import ConstantMasker
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from captum.attr import Saliency
import os
import torch.multiprocessing as mp


def deletion_demo(rank, world_size, x_batch, y_batch, queue, event):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = get_model().to(rank)
    x_batch = x_batch.to(rank)
    y_batch = y_batch.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    a_batch = Saliency(ddp_model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(dim=1, keepdims=True).cpu().numpy()

    masker = ConstantMasker(feature_level="pixel")
    del_results = deletion(x_batch, y_batch, ddp_model, a_batch, masker)
    print(rank, del_results)
    queue.put((rank, del_results))
    event.wait()

    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    mp.set_start_method("spawn")
    queue = mp.Queue()  # For storing the results
    event = mp.Event()  # Used for signaling processes that they can stop (once all data is read)
    processes = []  # Used for joining all processes later
    world_size = torch.cuda.device_count()

    dataloader = get_dataloader()
    for rank in range(world_size):
        x_batch, y_batch = iter(dataloader).next()
        p = mp.Process(target=deletion_demo, args=(rank, world_size, x_batch, y_batch, queue, event))
        p.start()
        processes.append(p)

    for _ in range(world_size):
        res = queue.get()
        print("Main process received:", res, flush=True)
    event.set()

    for p in processes:
        p.join()
