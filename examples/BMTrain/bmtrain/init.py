import datetime
import torch
import random
import torch.distributed as dist
import os
from .utils import print_dict
import ctypes
from .global_var import config

from . import nccl
from .synchronize import synchronize


def init_distributed(
    init_method: str = "env://",
    seed: int = 0,
    pipe_size: int = -1,
    num_micro_batches: int = None,
    tp_size: int = 1,
):
    """Initialize distributed training.
    This function will initialize the distributed training, set the random seed and global configurations.
    It must be called before any other distributed functions.

    Args:
        seed (int): The random seed.
        pipe_size (int) : pipe_size means that all processes will be divided into pipe_size groups
        num_micro_batches (int) : means that the input batchs will be divided into num_micro_batches small batches. used in pipeline mode.
        tp_size (int) : tp_size means the size of each of tensor parallel group

    **init_distributed** reads the following environment variables:

    * `WORLD_SIZE`: The total number gpus in the distributed training.
    * `RANK`: The global rank of the current gpu. From 0 to `WORLD_SIZE - 1`.
    * `MASTER_ADDR`: The address of the master node.
    * `MASTER_PORT`: The port of the master node.
    * `LOCAL_RANK`: The local rank of the current gpu.

    Normally, all the environments variables above are setted by the pytorch distributed launcher.

    **Note**: Do not use any functions in torch.distributed package including `torch.distributed.init_process_group` .

    **Note**: If your training script is stuck here , it means some of your distributed workers are not connected to the master node.

    """
    torch.backends.cudnn.enabled = False

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "10010"
    addr = os.environ["MASTER_ADDR"]
    port = os.environ["MASTER_PORT"]
    master = addr + ":" + port
    timeout = datetime.timedelta(seconds=1800)
    rendezvous_iterator = dist.rendezvous(
        init_method, rank, world_size, timeout=timeout
    )

    store, rank, world_size = next(rendezvous_iterator)
    store.set_timeout(timeout)
    store = dist.PrefixStore("bmtrain", store)
    torch.cuda.set_device(local_rank)
    config["initialized"] = True
    config["pipe_size"] = pipe_size if pipe_size > 0 else 1
    config["pipe_enabled"] = pipe_size > 0
    config["local_rank"] = local_rank
    config["local_size"] = local_size
    config["rank"] = rank
    config["world_size"] = world_size
    config["calc_stream"] = torch.cuda.current_stream()
    config["load_stream"] = torch.cuda.Stream(priority=-1)
    config["tp_comm_stream"] = torch.cuda.Stream(priority=-1)
    config["pp_comm_stream"] = torch.cuda.Stream(priority=-1)
    config["barrier_stream"] = torch.cuda.Stream()
    config["load_event"] = torch.cuda.Event()
    config["tp_size"] = tp_size if tp_size > 0 else 1
    config["topology"] = topology(config)
    config["zero_rank"] = config["topology"].get_group_rank("zero")
    config["tp_rank"] = config["topology"].get_group_rank("tp")
    config["tp_zero_rank"] = config["topology"].get_group_rank("tp_zero")
    config["save_param_to_cpu"] = True
    cpus_this_worker = None

    all_available_cpus = sorted(list(os.sched_getaffinity(0)))

    cpus_per_worker = len(all_available_cpus) // local_size

    if cpus_per_worker < 1:
        cpus_this_worker = all_available_cpus
        torch.set_num_threads(1)
    else:
        cpus_this_worker = all_available_cpus[
            local_rank * cpus_per_worker : (local_rank + 1) * cpus_per_worker
        ]
        os.sched_setaffinity(0, cpus_this_worker)
        torch.set_num_threads(len(cpus_this_worker))

    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

    if rank == 0:
        unique_id: bytes = nccl.getUniqueId()
        store.set("BMTRAIN_UNIQUE_ID", unique_id.hex())

    unique_id = bytes.fromhex(store.get("BMTRAIN_UNIQUE_ID").decode())
    config["comm"] = nccl.commInitRank(unique_id, world_size, rank)
    topo = config["topology"]

    if config["pipe_enabled"]:
        config["micros"] = (
            num_micro_batches if num_micro_batches else config["pipe_size"]
        )
        if topo.stage_id == 0:
            unique_id = nccl.getUniqueId()
            store.set(f"PIPE_UNIQUE_ID{topo.pipe_idx}", unique_id.hex())
        unique_id = bytes.fromhex(store.get(f"PIPE_UNIQUE_ID{topo.pipe_idx}").decode())
        config["pipe_comm"] = nccl.commInitRank(unique_id, pipe_size, topo.stage_id)

        if topo.pp_zero_id == 0:
            unique_id = nccl.getUniqueId()
            store.set(f"PP_ZERO_UNIQUE_ID{topo.pp_zero_idx}", unique_id.hex())
        unique_id = bytes.fromhex(
            store.get(f"PP_ZERO_UNIQUE_ID{topo.pp_zero_idx}").decode()
        )
        config["pp_zero_comm"] = nccl.commInitRank(
            unique_id, world_size // config["pipe_size"], topo.pp_zero_id
        )

    if config["tp_size"] > 1:
        if topo.tp_id == 0:
            unique_id = nccl.getUniqueId()
            store.set(f"TP_UNIQUE_ID{topo.tp_idx}", unique_id.hex())
        unique_id = bytes.fromhex(store.get(f"TP_UNIQUE_ID{topo.tp_idx}").decode())
        config["tp_comm"] = nccl.commInitRank(unique_id, tp_size, topo.tp_id)

        if topo.tp_zero_id == 0:
            unique_id = nccl.getUniqueId()
            store.set(f"TP_ZERO_UNIQUE_ID{topo.tp_zero_idx}", unique_id.hex())
        unique_id = bytes.fromhex(
            store.get(f"TP_ZERO_UNIQUE_ID{topo.tp_zero_idx}").decode()
        )
        config["tp_zero_comm"] = nccl.commInitRank(
            unique_id, world_size // config["tp_size"], topo.tp_zero_id
        )

    if config["pipe_size"] > 1 and config["tp_size"] > 1:
        if topo.pp_tp_zero_id == 0:
            unique_id = nccl.getUniqueId()
            store.set(f"PP_TP_ZERO_UNIQUE_ID{topo.pp_tp_zero_idx}", unique_id.hex())
        unique_id = bytes.fromhex(
            store.get(f"PP_TP_ZERO_UNIQUE_ID{topo.pp_tp_zero_idx}").decode()
        )
        config["pp_tp_zero_comm"] = nccl.commInitRank(
            unique_id,
            world_size // (config["pipe_size"] * config["tp_size"]),
            topo.pp_tp_zero_id,
        )

    config["zero_comm"] = config["comm"]

    for i in range(world_size):
        if i == rank:
            print_dict(
                "Initialization",
                {
                    "rank": rank,
                    "local_rank": local_rank,
                    "world_size": world_size,
                    "local_size": local_size,
                    "master": master,
                    "device": torch.cuda.current_device(),
                    "cpus": cpus_this_worker,
                },
            )
        synchronize()


class topology:
    """A helper class to keep parallel information when using different parallel methods together."""

    def __init__(self, config):
        # pipe_idx is the idx of the pipeline in the group
        self.rank = config["rank"]
        pp_size = config["pipe_size"]
        tp_size = config["tp_size"]
        world_size = config["world_size"]
        assert (
            world_size % (pp_size * tp_size) == 0
        ), "The nums of GPUs must be divisible by the pipeline parallel size * tensor parallel size"

        dp_size = world_size // (pp_size * tp_size)
        config["tp_zero_size"] = dp_size
        config["zero_size"] = world_size // pp_size
        self.stages = config["pipe_size"]

        stage_size = world_size // pp_size
        for i in range(world_size):
            self.pipe_idx = self.rank % stage_size
            self.stage_id = self.rank // stage_size
            self.tp_id = self.rank % tp_size
            self.tp_idx = self.rank // tp_size
            # pp->zero
            self.pp_zero_idx = self.stage_id
            self.pp_zero_id = self.pipe_idx
            # tp->zero
            self.tp_zero_idx = self.tp_id
            self.tp_zero_id = self.tp_idx
            # pp->tp->zero
            self.pp_tp_zero_idx = self.stage_id * tp_size + self.tp_id
            self.pp_tp_zero_id = self.pipe_idx // tp_size
        # only zero
        self.zero_idx = 0
        self.zero_id = self.rank

    def get_group_id(self, group_name):
        """Get group id of different parallel group.

        Args:
            group_name (str): must be one of "pipe", "zero", "tp_zero" or "tp".
        """
        if group_name == "pipe":
            return self.pipe_idx
        elif group_name == "zero":
            return self.zero_idx
        elif group_name == "tp_zero":
            return self.tp_zero_idx
        elif group_name == "tp":
            return self.tp_idx

    def get_group_rank(self, group_name):
        """Get group rank of different parallel group.

        Args:
            group_name (str): must be one of "pipe", "zero", "tp_zero" or "tp".
        """
        if group_name == "pipe":
            return self.stage_id
        elif group_name == "zero":
            return self.zero_id
        elif group_name == "tp_zero":
            return self.tp_zero_id
        elif group_name == "tp":
            return self.tp_id


def is_initialized() -> bool:
    return config["initialized"]
