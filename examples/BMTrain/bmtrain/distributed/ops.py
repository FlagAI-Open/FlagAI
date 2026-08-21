import torch
from ..global_var import config
from ..nccl import allGather as ncclAllGather, recv
from ..nccl import allReduce as ncclAllReduce
from ..nccl import broadcast as ncclBroadcast
from ..nccl import reduceScatter as ncclReduceScatter
from ..nccl import send as ncclSend
from ..nccl import recv as ncclRecv
from ..nccl import commCount,commRank,NCCLCommunicator
DTYPE_LIST = [
    torch.float64,
    torch.float32,
    torch.float16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.bfloat16,
    torch.bool
]
def send_activations(hidden_state, next_rank, comm):
    send_meta(hidden_state, next_rank, comm)
    ncclSend(hidden_state.storage(), next_rank, comm)

def recv_activations(prev_rank, comm):
    dtype, shape = recv_meta(prev_rank, comm)
    hidden_state = torch.empty(shape, dtype=dtype, device="cuda")
    ncclRecv(hidden_state.storage(), prev_rank, comm)
    return hidden_state

def send_meta(x, next_rank, comm):
    meta_data = torch.tensor(data=[0]*50, device="cuda", dtype=torch.int)
    meta_data[0] = len(x.size())
    meta_data[1] = DTYPE_LIST.index(x.dtype)
    meta_data[2:len(x.size())+2] = torch.tensor(x.size(), device="cuda", dtype=torch.int)
    meta_data = meta_data.contiguous()
    ncclSend(meta_data.storage(), next_rank, comm)

def recv_meta(prev_rank, comm):
    meta_data = torch.tensor(data=[0]*50, device="cuda", dtype=torch.int)
    ncclRecv(meta_data.storage(), prev_rank, comm)
    n_dims = meta_data[0].item()
    dtype = DTYPE_LIST[meta_data[1].item()]
    shape = meta_data[2:n_dims+2].tolist()
    return dtype,shape

class OpBroadcast(torch.autograd.Function):

    @staticmethod
    def forward(ctx, src, root, comm = None):
        if comm is None:
            comm = config["comm"]
        ctx.comm = comm
        outputs = torch.empty_like(src, dtype = src.dtype, device = src.device)
        ncclBroadcast(src.storage(), outputs.storage(), root, comm)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        res = all_reduce(grad_output, "sum", ctx.comm)
        return res, None, None

def broadcast(src, root, comm=None):
    if not config["initialized"]:
        raise RuntimeError("BMTrain is not initialized")
    return OpBroadcast.apply(src, root, comm)

class OpAllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input : torch.Tensor, comm = None):
        if comm is None:
            comm = config["comm"]
        world_size = commCount(comm)
        if not input.is_contiguous():
            input = input.contiguous()
        if input.storage_offset() != 0 or input.storage().size() != input.numel():
            input = input.clone()
        output = torch.empty( (world_size,) + input.size(), dtype=input.dtype, device=input.device)
        ctx.comm = comm
        ncclAllGather(
            input.storage(),
            output.storage(),
            comm
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output[commRank(ctx.comm)], None

def all_gather(x : torch.Tensor, comm = None):
    """Gathers the input tensor from all processes.

    Args:
        x (torch.Tensor): The input tensor of shape (...).
    
    Returns:
        torch.Tensor: The gathered tensor of shape (world_size, ...).
    """
    if not config["initialized"]:
        raise RuntimeError("BMTrain is not initialized")
    
    assert x.is_cuda
    return OpAllGather.apply(x, comm)

class OpReduceScatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input : torch.Tensor, op : str, comm : NCCLCommunicator = None):
        if comm is None:
            comm = config["comm"]
        ctx.comm = comm
        rank = commRank(comm)
        assert input.shape[0] % commCount(comm) == 0, "The dimension 0 must be divisible by the number of communication processes"
        if not input.is_contiguous():
            input = input.contiguous()
        if input.storage_offset() != 0 or input.storage().size() != input.numel():
            input = input.clone()
        output_shape = (input.shape[0] // commCount(comm), *input.shape[1:])
        output = torch.empty( output_shape, dtype=input.dtype, device=input.device )
        ncclReduceScatter(
            input.storage(),
            output.storage(),
            op,
            comm
        )
        ctx.op = op
        if op in ["sum", "avg"]:
            pass
        elif op in ["max", "min"]:
            ctx.save_for_backward( output != input[rank * input.shape[0]:(rank + 1) * input.shape[0]] )
        else:
            ctx.save_for_backward( output / input[rank * input.shape[0]:(rank + 1) * input.shape[0]] )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            grad_output = OpAllGather.apply(grad_output, ctx.comm).flatten(0,1)
        if ctx.op in ["max", "min", "prod"]:
            raise NotImplementedError("max min operation now do not support backward")
        else:
            if ctx.op == "avg":
                grad_output /= commCount(ctx.comm)
            return grad_output, None, None
       

def reduce_scatter(x : torch.Tensor, op : str = "sum", comm = None):
    """Reduces the input tensor from all processes.

    Args:
        x (torch.Tensor): The input tensor of shape (world_size, ...).
        op (str): The reduction operation, one of "sum", "avg", "max", "min", "prod". Default: "sum".

    Returns:
        torch.Tensor: The reduced tensor of shape (...).
    
    """
    if not config["initialized"]:
        raise RuntimeError("BMTrain is not initialized")

    assert x.is_cuda
    return OpReduceScatter.apply(x, op, comm)

class OpAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input : torch.Tensor, op : str, comm : NCCLCommunicator = None):
        if comm is None:
            comm = config["comm"]
        ctx.comm = comm
        if not input.is_contiguous():
            input = input.contiguous()
        if input.storage_offset() != 0 or input.storage().size() != input.numel():
            input = input.clone()
        output = torch.empty( input.size(), dtype=input.dtype, device=input.device)
        
        ncclAllReduce(
            input.storage(),
            output.storage(),
            op,
            comm
        )
        ctx.op = op
        
        if op in ["sum", "avg"]:
            pass
        elif op in ["max", "min"]:
            ctx.save_for_backward( input != output )
        else:
            ctx.save_for_backward( output / input )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.op == "sum":
            return grad_output, None, None
        elif ctx.op == "avg":
            return grad_output / commCount(ctx.comm), None, None
        elif ctx.op in ["max", "min"]:
            return torch.masked_fill(grad_output, ctx.saved_tensors[0], 0), None, None
        else:
            return grad_output * ctx.saved_tensors[0], None, None

def all_reduce(x : torch.Tensor, op : str = "sum", comm = None):
    """Reduces the input tensor from all processes.

    Args:
        x (torch.Tensor): The input tensor of shape (...).
        op (str): The reduction operation, one of "sum", "avg", "max", "min", "prod". Default: "sum".

    Returns:
        torch.Tensor: The reduced tensor of shape (...).
    
    """
    if not config["initialized"]:
        raise RuntimeError("BMTrain is not initialized")

    assert x.is_cuda
    return OpAllReduce.apply(x, op, comm)


            
