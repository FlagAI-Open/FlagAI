import torch
from . import distributed, nccl
from .global_var import config
import warnings

def synchronize():
    """
    Synchronize all the workers across all nodes. (both CPU and GPU are synchronized)
    """
    if not config["initialized"]:
        raise RuntimeError("BMTrain is not initialized")

    with torch.cuda.stream(config['barrier_stream']):
        barrier = torch.cuda.FloatTensor([1])
        nccl.allReduce(barrier.storage(), barrier.storage(), 'sum', config['comm'])
    config['barrier_stream'].synchronize()

def wait_loader():
    if not config["initialized"]:
        raise RuntimeError("BMTrain is not initialized")

    # wait lastest loader event, and set a new one
    config['load_event'].synchronize()
    config['calc_stream'].record_event(config['load_event'])

def wait_all_stream():
    streams = [config[s] for s in ["calc_stream", "load_stream", "offload_stream", "prefetch_stream"]]
    for s1 in streams:
        for s2 in streams:
            s1.wait_stream(s2)

def sum_loss(loss : torch.Tensor):
    """
    Sum the loss across all workers.

    This is a helper function to reduce the loss across all workers.
    """
    warnings.warn("bmtrain.sum_loss is deprecated and will be removed in later version. Use bmtrain.distributed.all_reduce instead.", DeprecationWarning)
    return distributed.all_reduce(loss, "sum") / config['world_size']

def gather_result(result: torch.Tensor):
    warnings.warn("bmtrain.gather_result is deprecated and will be removed in later version. Use bmtrain.distributed.all_gather instead.", DeprecationWarning)

    output_cuda = True
    if not result.is_cuda:
        result = result.cuda()
        output_cuda = False
    ret = torch.empty((result.shape[0]*config['world_size'], *list(result.shape[1:])), device=result.device, dtype=result.dtype)
    nccl.allGather(
        result.storage(),
        ret.storage(),
        config['comm']
    )
    if output_cuda:
        return ret
    else:
        return ret.cpu()
