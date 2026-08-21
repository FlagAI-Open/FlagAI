
import torch
import torch.nn as nn
from typing import Optional
import opendelta.utils.logging as logging

logger = logging.get_logger(__name__)


def inspect_module_statistics(module: Optional[nn.Module]=None, verbose=True):
    r"""Get the statistics of the parameters in the delta modules.

    Args:
        module (:obj:`nn.Module`, *optional*): The module to compute the statistics.

    Returns:
        :obj:`dict`: The statistics of the parameters in the delta modules.

    """

    stat = {}
    n_trainable = num_trainable_parameters(module)
    n_total = num_total_parameters(module)

    stat['total_parameters'] = n_total
    stat['trainable_parameters'] = n_trainable

    stat['trainable_ratio'] = n_trainable/n_total

    n_delta = num_delta_parameters(module)
    n_total = num_total_parameters(module)
    stat['delta_parameters'] = n_delta
    stat['delta_ratio'] = n_delta/n_total

    cudamem = 0
    maxcudamem = 0
    for device_id in range(torch.cuda.device_count()):
        cudamem += torch.cuda.memory_allocated(f"cuda:{device_id}")/1024**3
        maxcudamem += torch.cuda.max_memory_allocated(f"cuda:{device_id}")/1024**3
    stat['cudamem'] = cudamem
    stat['maxcudamem'] = maxcudamem

    if verbose:
        logger.info(stat)

    return stat

def num_trainable_parameters(module: Optional[nn.Module]=None):
    r"""[NODOC] A small sugar function to get the number of trainable parameter in the backbone model. Often used to
    compute the trainable rate.

    Args:
        module (:obj:`nn.Module`): of which module we want to know the number of trainable paramemters.

    Returns:
        :obj:`List[nn.Parameter]`
    """
    pnum_tot = 0
    for param in module.parameters():
        if param.requires_grad:
            pnum_tot += param.numel()
    return pnum_tot


def num_total_parameters(module: Optional[nn.Module]=None):
    r"""[NODOC] A small sugar function to get the number of trainable parameter in the backbone model. Often used to
    compute the trainable rate.

    Args:
        module (:obj:`nn.Module`): of which module we want to know the number of trainable paramemters.

    Returns:
        :obj:`List[nn.Parameter]`
    """
    pnum_tot = 0
    for param in module.parameters():
        pnum_tot += param.numel()
    return pnum_tot

def num_delta_parameters(module: Optional[nn.Module]=None):
    r"""[NODOC] A small sugar function to get the number of trainable parameter in the backbone model. Often used to
    compute the trainable rate.

    Args:
        module (:obj:`nn.Module`): of which module we want to know the number of trainable paramemters.

    Returns:
        :obj:`List[nn.Parameter]`
    """
    pnum_tot = 0
    for param in module.parameters():
        if hasattr(param, "_is_delta"):
            pnum_tot += param.numel()
    return pnum_tot

def inspect_optimizer_statistics(optimizer, verbose=True):
    stats = {}
    for id, param_group in enumerate(optimizer.param_groups):
        stat = {}
        fine_grain_info = [(p.numel(), p.requires_grad) for p in param_group['params']]
        stat['total_parameters'] = sum(n for n, r in fine_grain_info)
        stat['trainable_parameters'] = sum(n for n, r in fine_grain_info if r)
        stat['trainable_ratio'] = "{:.6f}%".format(stat['trainable_parameters']/stat['total_parameters']*100)
        for key in param_group:
            if key != 'params':
                stat[key] = param_group[key]
        stats[f'param_group_{id}'] = stat

    if verbose:
        logger.info(f"optimizer info: {stats}")

    return stat
