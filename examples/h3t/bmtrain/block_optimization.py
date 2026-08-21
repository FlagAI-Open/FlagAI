import torch
from typing_extensions import TypedDict

class TBLAutoOptimizationConfig(TypedDict):
    memory_limit : int
    algorithm : str
    kwargs : dict

def gen_tbl_optim_config(
    tbl_auto_optimization = None,
    tbl_memory_limit = None,
    tbl_optimization_kwargs = None
):
    if not tbl_auto_optimization:
        return None
    if tbl_memory_limit is None:
        raise ValueError("tbl_memory_limit required for TBL auto optimization.")
    alg_name = tbl_auto_optimization
    if tbl_optimization_kwargs is None:
        tbl_optimization_kwargs = {}
    return TBLAutoOptimizationConfig(
        memory_limit = tbl_memory_limit,
        algorithm = alg_name,
        kwargs = tbl_optimization_kwargs,
    )


class BlockOptimization(TypedDict):
    zero_level : int
    offload_parameter : bool
    checkpointing : bool
    offload_hidden_state : bool
    economical_forward : bool
    economical_backward : bool
    segment_synchronization : bool

def validate_boptim(self):
    # Check data type
    assert self["zero_level"] in [2, 3]
    self["offload_parameter"] = bool(self["offload_parameter"])
    self["checkpointing"] = bool(self["checkpointing"])
    self["offload_hidden_state"] = bool(self["offload_hidden_state"])
    if "economical_forward" not in self:
        self["economical_forward"] = False
    self["economical_forward"] = bool(self["economical_forward"])
    if "economical_backward" not in self:
        self["economical_backward"] = True
    self["economical_backward"] = bool(self["economical_backward"])
    if "segment_synchronization" not in self:
        self["segment_synchronization"] = True
    self["segment_synchronization"] = bool(self["segment_synchronization"])

    # Check conflicts
    if (not self["checkpointing"]) and self["offload_hidden_state"]:
        raise ValueError("Non-checkpointing conflicts with hidden state offloading.")

    return self


def max_block_optim():
    return BlockOptimization(
        zero_level = 3,
        offload_parameter = True,
        checkpointing = True,
        offload_hidden_state = True,
        economical_forward = True,
        economical_backward = True,
        segment_synchronization = True,
    )

def encode_block_optim(optim):
    ret = 0
    for v in [
        optim["zero_level"] - 2,
        optim["offload_parameter"],
        optim["checkpointing"],
        optim["offload_hidden_state"],
        optim["economical_forward"],
        optim["economical_backward"],
        optim["segment_synchronization"],
    ]:
        ret = ret * 2 + v
    return ret
