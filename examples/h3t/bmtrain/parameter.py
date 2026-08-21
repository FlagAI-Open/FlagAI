from typing import Callable, Iterable, Optional
import torch
from .utils import round_up
from .global_var import config
from . import nccl
import warnings

import threading

grad_accumulation_lock = threading.Lock()
def _grad_accumulation_cpu(dst, src, event):
    event.synchronize()
    grad_accumulation_lock.acquire()
    dst += src
    grad_accumulation_lock.release()



class DistributedParameter(torch.nn.Parameter):
    r"""
    DistributedParameter is a subclass of torch.nn.Parameter.

    It scatters the tensor to all the nodes and gathers them when needed.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient.
        init_method (Callable[['DistributedParameter'], None], optional): the method to initialize the parameter.
        group (str, optional): the group name of the parameter.

    **Note**: DistributedParameter must be on the CUDA device. It will transfer the data to device automatically when `__init__` called.

    """
    
    _original_shape : torch.Size
    _start_partition : int
    _end_partition : int
    _init_method : Optional[Callable[['DistributedParameter'], None]]
    _in_checkpoint_block : bool
    _group : Optional[str]

    def __new__(cls,
            data : torch.Tensor, 
            requires_grad : bool = True, 
            init_method : Optional[Callable[['DistributedParameter'], None]] = None,
            group : Optional[str] = None
        ):
        if not config["initialized"]:
            raise RuntimeError("BMTrain is not initialized")

        num_of_elements = data.numel()

        cuda_tensor = torch.tensor([], dtype=data.dtype, device="cuda") 
        cuda_storage_size = round_up(num_of_elements, config["world_size"]) // config["world_size"]

        original_shape = data.size()

        cuda_storage = cuda_tensor.storage_type()(cuda_storage_size)

        start_of_partition = cuda_storage_size * config["rank"]
        end_of_partition = min(num_of_elements, cuda_storage_size * (config["rank"] + 1))

        # FX: cuda_tensor_size < 0 if num_of_elements is too small
        cuda_tensor_size = max(end_of_partition - start_of_partition, 0)

        cuda_tensor.set_(cuda_storage, 0, (cuda_tensor_size,))
        cuda_tensor.copy_(data.view(-1)[start_of_partition: end_of_partition])
        ret = torch.Tensor._make_subclass(cls, cuda_tensor, requires_grad)
        
        setattr(ret, "_original_shape", original_shape)
        setattr(ret, "_start_partition", start_of_partition)
        setattr(ret, "_end_partition", end_of_partition)
        setattr(ret, "_init_method", init_method)
        setattr(ret, "_in_checkpoint_block", False)
        setattr(ret, "_group", group)
        setattr(ret, "_cpu_parameter", None)
        setattr(ret, "_on_host", False)
        setattr(ret, "_on_device", True)
        setattr(ret, "_event", torch.cuda.Event())
        setattr(ret, "_threads", [])
        setattr(ret, "_grad_zeroed", False)
        setattr(ret, "_optimizer_timer", None)
        return ret

    def _set_partition(self, start_of_partition, end_of_partition):
        setattr(self, "_start_partition", start_of_partition)
        setattr(self, "_end_partition", end_of_partition)
    
    @property
    def group(self):
        """The group name of the distributed parameter."""

        return self._group

    @property
    def cpu_parameter(self):
        self.join()
        return self._cpu_parameter
    
    @property
    def on_host(self):
        return self._on_host
    
    @property
    def on_device(self):
        return self._on_device

    @property
    def event(self):
        return self._event
    
    def allocate_cpu_storage(self):
        assert self._cpu_parameter is None
        assert not self.on_host
        data = self.data
        cpu_tensor = torch.empty(data.size(), dtype=data.dtype, pin_memory=True)
        cpu_tensor.copy_(self.data, non_blocking = False) # ?
        cpu_param = torch.nn.Parameter(cpu_tensor)
        if self.requires_grad:
            cpu_param.requires_grad_(True)
            cpu_param.grad = torch.empty(data.size(), dtype=data.dtype, pin_memory=True).zero_()
            self._grad_zeroed = True
        else:
            cpu_param.requires_grad_(False)
        setattr(self, "_cpu_parameter", cpu_param)
        setattr(self, "_on_host", True)

    def release_gpu_storage(self):
        # Only offloaded parameters support release_gpu_storage
        self.event.wait()
        assert self.on_host
        assert self.on_device
        self.data = torch.tensor([], dtype=self.dtype, device=self.device)
        self.grad = None
        setattr(self, "_on_device", False)
        self.event.record()

    def allocate_gpu_storage(self, storage = None, offset = 0):
        assert self.on_host
        assert not self.on_device
        cuda_tensor = torch.tensor([], dtype=self.dtype, device="cuda")
        cuda_tensor_size = self._cpu_parameter.size()
        if storage is None:
            cuda_storage_size = self._cpu_parameter.numel()
            cuda_storage = cuda_tensor.storage_type()(cuda_storage_size)
            cuda_tensor.set_(cuda_storage, 0, cuda_tensor_size)
        else:
            cuda_tensor.set_(storage, offset, cuda_tensor_size)
        self.data = cuda_tensor
        setattr(self, "_on_device", True)

    def prefetch(self, allocate_gpu_storage : bool, non_blocking : bool = False):
        self.event.wait()
        assert self.on_host
        if allocate_gpu_storage:
            self.allocate_gpu_storage()
        assert self.on_device
        assert self.data.shape == self._cpu_parameter.data.shape
        self.data.copy_(self._cpu_parameter.data, non_blocking = non_blocking)
        self.event.record()

    def offload(self, release_gpu_storage : bool, non_blocking : bool = False, grad = None):
        assert self.on_device
        assert self.on_host
        if grad is None:
            grad = self.grad
            specified_grad = False
        else:
            specified_grad = True
        if self.requires_grad and grad is not None:
            self.event.wait()
            self._copy_grad(self._cpu_parameter.grad, grad, self.event, non_blocking)
            self.event.record()
            self.event.wait()
            grad.record_stream(torch.cuda.current_stream())
        if release_gpu_storage:
            self.release_gpu_storage()
        elif not specified_grad:
            self.grad = None

    def _copy_grad(self, dst, src, event, non_blocking):
        if self._grad_zeroed:
            dst.copy_(src, non_blocking = non_blocking)
            self._grad_zeroed = False
        else:
            _src = torch.empty(src.size(), dtype=src.dtype, pin_memory=True)
            _src.copy_(src, non_blocking = non_blocking)
            event.record()
            t = threading.Thread(target=_grad_accumulation_cpu, args=(dst, _src, event))
            t.start()
            self._threads.append(t)

    def join(self):
        for t in self._threads:
            t.join()
        setattr(self, "_threads", [])

    def gather(self) -> torch.Tensor:
        """Gather the data from all the distributed nodes.

        Return:
            torch.Tensor: The gathered data.
        
        """
        with torch.cuda.stream(config['load_stream']):
            output_tensor = OpAllGather.apply(self)
        current_stream = torch.cuda.current_stream()
        output_tensor.record_stream( current_stream )
        current_stream.wait_stream(config['load_stream'])
        return output_tensor

    def _copy_data(self, data : torch.Tensor):
        if self.on_device:
            self.data.copy_(data.view(-1)[self._start_partition : self._end_partition])
        if self.on_host:
            self.cpu_parameter.data.copy_(data.view(-1)[self._start_partition : self._end_partition])
    

class OpAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value : DistributedParameter):
        assert isinstance(value, DistributedParameter)

        # TODO : prefetch before gathering & offload after scattering
        if not value.on_device:
            assert value.on_host
            value.prefetch(allocate_gpu_storage = True)
            ctx.release_gpu_after_backward = True
        else:
            ctx.release_gpu_after_backward = False

        partition_size = value.storage().size()
        global_size = partition_size * config['world_size']

        storage = value.storage_type()(global_size)
        
        nccl.allGather(
            value.storage(),
            storage,
            config['comm']
        )

        output_tensor = torch.tensor([], dtype=value.dtype, device="cuda")
        output_tensor.set_(storage, 0, value._original_shape)
    
        ctx.partition_size = partition_size
        ctx.tensor_size = value.size(0)
        ctx.value = value
        return output_tensor
    
    @staticmethod
    def backward(ctx, grad_output : torch.Tensor):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        grad_storage = grad_output.storage_type()(ctx.partition_size)
        grad_output_storage = grad_output.storage()
        if grad_output_storage.size() == ctx.partition_size * config['world_size']:
            pass
        else:
            grad_output_storage.resize_(ctx.partition_size * config['world_size'])
        nccl.reduceScatter(
            grad_output_storage,
            grad_storage,
            'sum',
            config['comm']
        )
        grad_tensor = torch.tensor([], dtype=grad_output.dtype, device="cuda")
        grad_tensor.set_(grad_storage, 0, (ctx.tensor_size,))
        
        value = ctx.value
        if value.on_host:
            if ctx.release_gpu_after_backward:
                value.offload(release_gpu_storage = True, grad = grad_tensor)
                return None
            else:
                value.offload(release_gpu_storage = False, grad = grad_tensor)
                return grad_tensor

        return grad_tensor

class ParameterInitializer:
    """
    ParameterInitializer is a helper class that is used to initialize the distributed parameters.

    Similar to functools.partial .

    """
    def __init__(self, func : Callable, *args, **kwargs) -> None:
        self.func = func
        self._args = args
        self._kwargs = kwargs
    
    def __call__(self, param : DistributedParameter):
        self.func(param, *self._args, **self._kwargs)
