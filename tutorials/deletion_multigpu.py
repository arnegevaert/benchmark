from util.get_dataset_model import get_dataloader, get_model
from attrbench.metrics import deletion
from attrbench.lib.masking import ConstantMasker
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from captum.attr import Saliency
import os
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def deletion_demo(rank, world_size):
    setup(rank, world_size)

    model = get_model().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    a_batch = Saliency(ddp_model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(dim=1, keepdims=True).cpu().numpy()

    masker = ConstantMasker(feature_level="pixel")
    del_results = deletion(x_batch, y_batch, ddp_model, a_batch, masker)

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    dataloader = get_dataloader()
    x_batch, y_batch = iter(dataloader).next()
    run_demo(deletion_demo, world_size=1)
