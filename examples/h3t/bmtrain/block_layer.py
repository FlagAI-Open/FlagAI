from typing import Dict, Iterable, Iterator, Union, List

from .utils import round_up
from .global_var import config
import torch
from . import nccl
from .synchronize import wait_loader, wait_all_stream
from .parameter import DistributedParameter, OpAllGather
from .checkpointing import ScopedTensorInspectorContext
from . import debug
from .distributed import all_gather
import copy
import time
import warnings
from .blist_optimization import TBLAutoOptimization

def _prefetch_hidden_state(src, profile, event = None, dst = None):
    config["prefetch_stream"].wait_stream(config["offload_stream"])
    with torch.cuda.stream(config["prefetch_stream"]), profile.timer("prefetch_hidden_state"):
        if event is None:
            event = torch.cuda.Event()
        if dst is None:
            dst = torch.empty(src.size(), dtype=src.dtype, device="cuda")
        dst.data.copy_(src.data, non_blocking = True)
        event.record()
        event.wait()
    config["offload_stream"].wait_stream(config["prefetch_stream"])
    config["offload_stream"].wait_event(event)
    return dst

def _offload_hidden_state(src, profile, event = None, dst = None):
    config["offload_stream"].wait_stream(config["prefetch_stream"])
    config["offload_stream"].wait_stream(config["calc_stream"])
    with torch.cuda.stream(config["offload_stream"]), profile.timer("offload_hidden_state"):
        if event is None:
            event = torch.cuda.Event()
        if dst is None:
            dst = torch.empty(src.size(), dtype=src.dtype, pin_memory=True)
        src = src.detach()
        dst.data.copy_(src.data, non_blocking = True)
        event.record()
        event.wait()
    src.record_stream(config["offload_stream"])
    config["prefetch_stream"].wait_stream(config["offload_stream"])
    config["prefetch_stream"].wait_event(event)
    return dst

# the flag is used to control the zero level , 0 means normal zero3 , 1 means forward without release parameter ,2 means backward without gather parameter
class OpCheckpointBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, placeholder, block : 'CheckpointBlock', preserve_rng_state, len_args, is_train, *args):
        if is_train:
            ctx.block = block
            ctx.preserve_rng_state = preserve_rng_state
            ctx.cuda_rng_state = torch.cuda.get_rng_state() if preserve_rng_state else None
        tensors = []
        others = []
        tensor_offloaded = []
        tensor_requires_grad = []
        _args, args = args, []
        for arg in _args:
            if torch.is_tensor(arg):
                req_grad = arg.requires_grad
                arg = arg.detach().requires_grad_(req_grad)
            args.append(arg)
        for arg in args:
            if torch.is_tensor(arg):
                tensor_requires_grad.append(arg.requires_grad)
                if arg.is_cuda and is_train and block.offload_hidden_state:
                    arg = _offload_hidden_state(arg, block.profile)
                    tensor_offloaded.append(True)
                else:
                    tensor_offloaded.append(False)
                tensors.append(arg)
                others.append(None)
            else:
                tensors.append(None)
                others.append(arg)
                tensor_offloaded.append(False)
                tensor_requires_grad.append(False)

        if is_train:
            ctx.nontensor_inputs = others
            ctx.tensor_offloaded = tensor_offloaded
            ctx.tensor_requires_grad = tensor_requires_grad
            ctx.len_args = len_args
            ctx.save_for_backward(*tensors)
        ctx.param_dict={}
        if block.zero_level == 2 and is_train:
            flag = 1
        else:
            flag = 0
        block_ctx = CheckpointBlockContext(block, ctx.param_dict, flag)
        with torch.no_grad():
            block_ctx.enter()

        curr_block_requires_grad = is_train and not block.checkpointing
        grad_context = torch.enable_grad() if curr_block_requires_grad else torch.no_grad()
        with grad_context, ScopedTensorInspectorContext() as inspector:
            inp_args = args[:len_args]
            inp_kwargs = {}
            for k, v in zip(args[len_args::2], args[len_args + 1::2]):
                inp_kwargs[k] = v
            outputs = block._module._call_impl(*inp_args, **inp_kwargs)
        for it in inspector.hidden_states:
            debug.append("_inspect_hidden_states", it)

        with torch.no_grad():
            block_ctx.exit()
        if is_train:
            ctx.inspect_list = inspector.hidden_states
            if not block.checkpointing:
                ctx.outputs = outputs
        return outputs.detach()

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")

        all_inputs = []
        input_reqires_grad = []
        len_args = ctx.len_args
        for tensor, other, offloaded, req_grad in zip(ctx.saved_tensors, ctx.nontensor_inputs, ctx.tensor_offloaded, ctx.tensor_requires_grad):
            if tensor is None:
                all_inputs.append(other)
                input_reqires_grad.append(False)
            else:
                input_reqires_grad.append(req_grad)
                if offloaded:
                    tensor = _prefetch_hidden_state(tensor, ctx.block.profile)
                    tensor.requires_grad = req_grad
                all_inputs.append(tensor)

        
        with torch.random.fork_rng(devices=[torch.cuda.current_device()], enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.cuda.set_rng_state(ctx.cuda_rng_state)
                if ctx.block.zero_level == 2:
                    flag = 2
                else:
                    flag = 0
            with torch.enable_grad(), ScopedTensorInspectorContext() as inspector, CheckpointBlockContext(ctx.block, ctx.param_dict, flag):
                inp_args = all_inputs[:len_args]
                inp_kwargs = {}
                for k, v in zip(all_inputs[len_args::2], all_inputs[len_args + 1::2]):
                    inp_kwargs[k] = v
                if ctx.block.checkpointing:
                    outputs = ctx.block._module._call_impl(*inp_args, **inp_kwargs)
                else:
                    outputs = ctx.outputs
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
    
                assert len(outputs) == len(grad_outputs)

                outputs_with_grad = []
                grad_of_output = []
                for i, output in enumerate(outputs):
                    if torch.is_tensor(output) and output.requires_grad:
                        outputs_with_grad.append(output)
                        grad_of_output.append(grad_outputs[i])

                # calculate gradients for inputs, also for parameters
                torch.autograd.backward(
                    outputs_with_grad,
                    grad_of_output,
                )
            assert len(ctx.inspect_list) == len(inspector.hidden_states), "Backward step changed"
            for i, it in enumerate(inspector.hidden_states):
                assert it["name"] == ctx.inspect_list[i]["name"], "Backward step changed"
                assert it["shape"] == ctx.inspect_list[i]["shape"], "Backward step changed"
                assert it["group"] == ctx.inspect_list[i]["group"], "Backward step changed"
                
                # change the tensor in placeholder
                ctx.inspect_list[i]["tensor"] = it["tensor"]

        grads = []
        for inp, requires_grad in zip(all_inputs, input_reqires_grad):
            if requires_grad:
                grads.append(inp.grad)
            else:
                grads.append(None)
        return (None, None, None, None, None) + tuple(grads)

class CheckpointBlockContext:
    def __init__(self, block : 'CheckpointBlock', ctx_dict : dict = None, flag : int = 0, pipe = False) -> None:
        self.block = block
        self.ctx_dict = ctx_dict
        self._param_buffer = {}
        self._grad_buffer = {}
        self._param_tensor = {}
        self._grad_tensor = {}
        self.flag = flag
        self.gather_event = torch.cuda.Event()
        self._need_release = False
        self._grad_ready = False
        self._param_ready = False
        if pipe:
            self.comm = config["zero_comm"] 
        else:
            self.comm = config["comm"]

    def enter_prefetch(self):
        """
        prefetch parameters
        """
        if self.flag == 2:
            return
        if not self.block.offload_parameter:
            assert self.block._on_device
        if self.block._on_device:
            return
        self.block._on_device = True
        config["prefetch_stream"].wait_stream(config["offload_stream"])
        with torch.cuda.stream(config["prefetch_stream"]), self.block.profile.timer("prefetch_parameter"):
            self.block.init_storage_buffers(offload = False)
            for param in self.block._param_info:
                if param["in_this_partition"]:
                    offset = param["begin"]
                    kw_name = param["kw_name"]
                    storage = self.block._storage_params[kw_name].storage()
                    param["parameter"].allocate_gpu_storage(storage = storage, offset = offset)
            for param in self.block._param_info:
                if param["in_this_partition"]:
                    param["parameter"].prefetch(allocate_gpu_storage = False, non_blocking = True)

    def enter_gather(self):
        """
        gather parameters
        """
        if self.block._ready:
            return
        self.block._ready = True
        self._need_release = True

        # wait_loader()
        # config["load_stream"].wait_stream(config["prefetch_stream"])
        with torch.cuda.stream(config["load_stream"]), self.block.profile.timer("gather_parameter"):
            assert self.block._on_device
            for param in self.block._param_info:
                param["parameter"].event.wait()
            for kw, val in self.block._storage_info.items():
                assert self.block._storage_params[kw].is_cuda
                assert kw not in self._param_buffer

                local_param = self.block._storage_params[kw]
                storage_type = local_param.storage_type()
                if self.flag != 2:
                    self._param_buffer[kw] = storage_type(val["partition_size"] * val["world_size"])
                    self._param_tensor[kw] = torch.tensor([], dtype=self._param_buffer[kw].dtype, device=self._param_buffer[kw].device).set_(self._param_buffer[kw])

            if self.flag != 2:
                nccl.groupStart()
                for kw, val in self.block._storage_info.items():
                    nccl.allGather(
                        self.block._storage_params[kw].storage(),
                        self._param_buffer[kw],
                        self.comm
                    )
                nccl.groupEnd()
            for param in self.block._param_info:
                param["parameter"].event.record()

            self.gather_event.record()
    
    def enter_build_param(self):
        self.gather_event.wait()
        current_stream = torch.cuda.current_stream()

        if not self._param_ready:
            self._param_ready = True
            # set wait stream for each storage
            for kw in self.block._storage_info.keys():
                if self.flag != 2:
                    self._param_tensor[kw].record_stream(current_stream)

            # update parameters in block
            for param in self.block._param_info:
                kw_name = param["kw_name"]
                offset = param["offset"]
                shape = param["shape"]

                if self.flag != 2:
                    dtype = self._param_buffer[kw_name].dtype
                    device = self._param_buffer[kw_name].device
                    param["parameter"].data = torch.tensor([], dtype=dtype, device=device).set_(self._param_buffer[kw_name], offset, shape)                
                else:
                    dtype = param["parameter"].data.dtype
                    device = param["parameter"].data.device
                    param["parameter"].data = torch.tensor([], dtype=dtype, device=device).set_(self.ctx_dict[kw_name], offset, shape)

        requires_grad = torch.is_grad_enabled()
        if requires_grad and not self._grad_ready:
            self._grad_ready = True
            for kw, val in self.block._storage_info.items():
                assert kw not in self._grad_buffer

                local_param = self.block._storage_params[kw]
                storage_type = local_param.storage_type()
                if local_param.requires_grad:
                    self._grad_buffer[kw] = storage_type(val["partition_size"] * val["world_size"])
                    self._grad_tensor[kw] = torch.tensor([], dtype=self._grad_buffer[kw].dtype, device=self._grad_buffer[kw].device).set_(self._grad_buffer[kw]).zero_()
                if kw in self._grad_tensor:
                    self._grad_tensor[kw].record_stream(current_stream)

            for param in self.block._param_info:
                kw_name = param["kw_name"]
                offset = param["offset"]
                shape = param["shape"]

                if self.flag != 2:
                    dtype = self._param_buffer[kw_name].dtype
                    device = self._param_buffer[kw_name].device
                else:
                    dtype = param["parameter"].data.dtype
                    device = param["parameter"].data.device
                if kw_name in self._grad_buffer and param["parameter"].requires_grad:
                    param["parameter"].grad = torch.tensor([], dtype=dtype, device=device).set_(self._grad_buffer[kw_name], offset, shape)

    def enter(self):
        self.enter_prefetch()
        self.enter_gather()
        self.enter_build_param()

    def __enter__(self):
        self.enter()
    
    def exit_offload(self):
        """
        offload gradients
        """
        if self.flag == 1:
            return
        requires_grad = torch.is_grad_enabled()
        # config["offload_stream"].wait_stream(config["load_stream"])
        config["offload_stream"].wait_stream(config["prefetch_stream"])
        with torch.cuda.stream(config["offload_stream"]), self.block.profile.timer("offload_gradient"):
            assert self.block._on_device
            for param in self.block._param_info:
                if not param["in_this_partition"]:
                    continue
                param = param["parameter"]
                if not param.on_host:
                    param.allocate_cpu_storage()
                if requires_grad and param.requires_grad:
                    param.offload(release_gpu_storage = False, non_blocking = True)
                    param._optimizer_timer = self.block.profile.optimizer_timer()
            if self.block.offload_parameter:
                for param in self.block._param_info:
                    if param["in_this_partition"]:
                        param["parameter"].release_gpu_storage()
                self.block._on_device = False
                self.block.init_storage_buffers(offload = True)
            else:
                for kw in self.block._storage_params.keys():
                    self.block._storage_params[kw].grad = None


    def exit_scatter(self):
        """
        Reduce scatter gradients
        """

        if not self._need_release:
            return
        self._need_release = False
        self.block._ready = False
        self._grad_ready = False
        self._param_ready = False
        requires_grad = torch.is_grad_enabled()
        if requires_grad:
            for kw, val in self.block._storage_info.items():
                local_param = self.block._storage_params[kw]

                # accumulate previous gradient
                if local_param.requires_grad:
                    if local_param.grad is None:
                        grad_storage = val["storage_type"](val["partition_size"])   # initialize gradient if not exist
                        local_param.grad = torch.tensor([], dtype=grad_storage.dtype, device=grad_storage.device).set_(grad_storage).zero_()
                    else:
                        self._grad_tensor[kw][val["begin"]:val["end"]] += local_param.grad
            
            current_stream = torch.cuda.current_stream()
            config["load_stream"].wait_stream(current_stream)   # wait for backward

            with torch.cuda.stream(config["load_stream"]), self.block.profile.timer("scatter_gradient"):
                nccl.groupStart()
                for kw, val in self.block._storage_info.items():
                    local_param = self.block._storage_params[kw]

                    # scatter gradient
                    if local_param.requires_grad:
                        nccl.reduceScatter(
                            self._grad_buffer[kw],
                            local_param.grad.storage(),
                            "sum",
                            self.comm
                        )
                nccl.groupEnd()
                for param in self.block._param_info:
                    param["parameter"].event.record()


            # set wait stream for each storage
            for kw in self._grad_tensor.keys():
                # grads can not be freed until reduce ops finish
                self._grad_tensor[kw].record_stream(config["load_stream"])

        # Release all parameters from buffer to block_storge
        for param in self.block._param_info:
            kw_name = param["kw_name"]
            dtype = self.block._storage_params[kw_name].dtype
            device = self.block._storage_params[kw_name].device
            if not param["in_this_partition"]:
                param["parameter"].data = torch.tensor([], dtype=dtype, device=device)
                param["parameter"].grad = None
                continue
            begin = param["begin"]
            end = param["end"]
            param["parameter"].data = torch.tensor([], dtype=dtype, device=device).set_(self.block._storage_params[kw_name].storage(), begin, end)
            if param["parameter"].requires_grad and self.block._storage_params[kw_name].grad is not None:
                param["parameter"].grad = torch.tensor([], dtype=dtype, device=device).set_(self.block._storage_params[kw_name].grad.storage(), begin, end)
        if self.flag == 1:
            for i in self._param_buffer:
                self.ctx_dict[i] = self._param_buffer[i]
        else:
            for i in self.ctx_dict:
                self.ctx_dict[i] = None

        self._grad_tensor = {}
        self._param_tensor = {}
        self._grad_buffer = {}
        self._param_buffer = {}
        config["load_stream"].record_event(config["load_event"])

    def exit(self):
        # reduce scatter gradients
        self.exit_scatter()
        self.exit_offload()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()

def dtype2memory(dtype):
    DTYPE_MEMORY_MAP = {
        torch.float32: 4,
        torch.float64: 8,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
    }
    if dtype not in DTYPE_MEMORY_MAP:
        raise ValueError("Unknown dtype: {}".format(dtype))
    return DTYPE_MEMORY_MAP[dtype]

def _storage_type(storage_type, to):
    to = to.lower().strip()
    assert to in ["cpu", "cuda"]
    STORAGE_MAP_CPU_TO_CUDA = {
        torch.FloatStorage: torch.cuda.FloatStorage,
        torch.DoubleStorage: torch.cuda.DoubleStorage,
        torch.HalfStorage: torch.cuda.HalfStorage,
        torch.BFloat16Storage: torch.cuda.BFloat16Storage,
        torch.CharStorage: torch.cuda.CharStorage,
        torch.ByteStorage: torch.cuda.ByteStorage,
        torch.ShortStorage: torch.cuda.ShortStorage,
        torch.IntStorage: torch.cuda.IntStorage,
    }
    STORAGE_MAP_CUDA_TO_CPU = {
        torch.cuda.FloatStorage: torch.FloatStorage,
        torch.cuda.DoubleStorage: torch.DoubleStorage,
        torch.cuda.HalfStorage: torch.HalfStorage,
        torch.cuda.BFloat16Storage: torch.BFloat16Storage,
        torch.cuda.CharStorage: torch.CharStorage,
        torch.cuda.ByteStorage: torch.ByteStorage,
        torch.cuda.ShortStorage: torch.ShortStorage,
        torch.cuda.IntStorage: torch.IntStorage,
    }
    if (storage_type not in STORAGE_MAP_CPU_TO_CUDA) and (storage_type not in STORAGE_MAP_CUDA_TO_CPU):
        raise ValueError("Unknown storage type: {}".format(storage_type))
    if (to == "cuda") and (storage_type in STORAGE_MAP_CPU_TO_CUDA):
        return STORAGE_MAP_CPU_TO_CUDA[storage_type]
    if (to == "cpu") and (storage_type in STORAGE_MAP_CUDA_TO_CPU):
        return STORAGE_MAP_CUDA_TO_CPU[storage_type]
    return storage_type

def storage_type_cuda(storage_type):
    return _storage_type(storage_type, to = "cuda")

def storage_type_cpu(storage_type):
    return _storage_type(storage_type, to = "cpu")

def _get_param_kw(param : DistributedParameter):
    type_name = str(param.dtype).split(".")[-1]
    grad_name = "_grad" if param.requires_grad else "_nograd"
    group_name = ""
    if param.group is not None:
        group_name = "_g_" + param.group
    return type_name + grad_name + group_name

class ModelProfile:
    class RuntimeProfile:
        def __init__(self, runtime_wb_func = None, memory_wb_func = None):
            self.writeback_runtime = runtime_wb_func
            self.writeback_memory = memory_wb_func
        def __enter__(self):
            self.enter()
        def enter(self):
            if self.writeback_memory is not None:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                self._start_mem = torch.cuda.memory_allocated()
            if self.writeback_runtime is not None:
                torch.cuda.synchronize()
                self._start_time = time.time()
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exit()
        def exit(self):
            if self.writeback_runtime is not None:
                torch.cuda.synchronize()
                _time = time.time() - self._start_time
                self.writeback_runtime(_time)
            if self.writeback_memory is not None:
                torch.cuda.synchronize()
                _delta_mem = torch.cuda.memory_allocated() - self._start_mem
                _peak_mem = torch.cuda.max_memory_allocated() - self._start_mem
                self.writeback_memory(delta = _delta_mem, peak = _peak_mem)
            
    def __init__(self, capacity = 100):
        self.capacity = capacity
        self.runtime = {
            "optimizer_step": [],
            "forward_no_grad": [],
            "forward_with_grad": [],
            "backward": [],
            "prefetch_hidden_state": [],
            "offload_hidden_state": [],
            "prefetch_parameter": [],
            "gather_parameter": [],
            "scatter_gradient": [],
            "offload_gradient": [],
        }
        self.curr_optimizer_runtime = []
        self.switch_on = True
        self.temp_on = True
        self.memory = {
            "input_tensor": 0,
        }
        self.convergent_keys = set()
    def input_recorder(self, input_tensor):
        if type(input_tensor) != list:
            input_tensor = [input_tensor]
        mem = 0
        for x in input_tensor:
            if torch.is_tensor(x) and x.is_cuda:
                mem += x.numel() * dtype2memory(x.dtype)
        if mem > self.memory["input_tensor"]:
            self.memory["input_tensor"] = mem
            self.runtime["forward_no_grad"] = []
            self.runtime["forward_with_grad"] = []
            self.runtime["backward"] = []
            self.runtime["offload_hidden_state"] = []
            self.runtime["prefetch_hidden_state"] = []
            self.temp_on = True
            self.memory["input_tensor"] = mem
        elif mem == self.memory["input_tensor"]:
            self.temp_on = True
            self.memory["input_tensor"] = mem
        else:
            self.temp_on = False

    def set_parameter_memory(self, partitioned_param, gathered_param, partitioned_grad, gathered_grad):
        self.memory["gathered_parameter"] = gathered_param
        self.memory["partitioned_parameter"] = partitioned_param
        self.memory["gathered_gradient"] = gathered_grad
        self.memory["partitioned_gradient"] = partitioned_grad
    def timer(self, key):
        if key not in self.runtime:
            self.runtime[key] = []
        elif len(self.runtime[key]) > self.capacity:
            self.runtime[key] = self.runtime[key][-self.capacity:]
        
        record_memory = ("forward" in key) or ("backward" in key)
        if self.switch_on and self.temp_on:
            def runtime_wb_func(runtime):
                self.runtime[key].append(runtime)
            if record_memory:
                def memory_wb_func(delta, peak):
                    for k, v in [(key + "_delta", delta), (key + "_peak", peak)]:
                        v = int(v)
                        if k in self.memory:
                            self.memory[k] = max(self.memory[k], v)
                        else:
                            self.memory[k] = v
            else:
                memory_wb_func = None
        else:
            runtime_wb_func = None
            memory_wb_func = None
        return ModelProfile.RuntimeProfile(
            runtime_wb_func = runtime_wb_func,
            memory_wb_func = memory_wb_func,
        )

    def optimizer_timer(self):
        if self.curr_optimizer_runtime:
            self.runtime["optimizer_step"].append(sum(self.curr_optimizer_runtime))
            if len(self.runtime["optimizer_step"]) > self.capacity:
                self.runtime["optimizer_step"] = self.runtime["optimizer_step"][-self.capacity:]
            self.curr_optimizer_runtime = []

        if self.switch_on:
            def runtime_wb_func(runtime):
                self.curr_optimizer_runtime.append(runtime)
        else:
            runtime_wb_func = None
        return ModelProfile.RuntimeProfile(runtime_wb_func = runtime_wb_func)
        
    def get_runtime(self, key = None, window_size = None, average = True):
        if key is None:
            ret = {}
            for key in self.runtime.keys():
                ret[key] = self.get_runtime(key = key, window_size = window_size, average = average)
            return ret
        self.all_gather_runtime(key = key)
        arr = self.runtime[key]
        if window_size is not None:
            assert window_size > 0
            arr = arr[-window_size:]
        if average:
            return float(torch.Tensor(arr).max(dim = 1).values.mean(dim = 0))
        return arr
    
    def convergence(self, window_size = 5, relative_error = 0.05):
        self.all_gather_runtime()
        ret = True
        for key in self.runtime.keys():
            if key in self.convergent_keys:
                continue
            if len(self.runtime[key]) < 2 * window_size:
                ret = False
                continue
            curr_window = self.runtime[key][-window_size:]
            prev_window = self.runtime[key][-2*window_size:-window_size]
            curr_result = float(torch.Tensor(curr_window).max(dim = 1).values.mean(dim = 0))
            prev_result = float(torch.Tensor(prev_window).max(dim = 1).values.mean(dim = 0))
            if abs(prev_result - curr_result) / curr_result < relative_error:
                self.convergent_keys.add(key)
                self.runtime[key] = self.runtime[key][-2*window_size:]
            else:
                ret = False
        return ret

    def all_gather_runtime(self, key = None):
        if key is None:
            for key in self.runtime.keys():
                self.all_gather_runtime(key = key)
            return
        arr = self.runtime[key]
        for i in range(len(arr)):
            if type(arr[i]) == list:
                continue
            t = torch.Tensor([arr[i]]).cuda()
            t = all_gather(t).cpu().view(-1).tolist()
            arr[i] = t

    def switch_off_(self):
        self.switch_on = False
    def switch_on_(self):
        self.switch_on = True
    
    def to_json(self):
        ret = {
            "capacity": self.capacity,
            "runtime": self.runtime,
            "memory": self.memory,
            "convergent_keys": list(self.convergent_keys),
        }
        return ret
    @staticmethod
    def from_json(obj):
        profile = ModelProfile(obj["capacity"])
        profile.runtime = obj["runtime"]
        profile.memory = obj["memory"]
        profile.convergent_keys = set(obj["convergent_keys"])
        return profile

class CheckpointBlock(torch.nn.Module):
    """ Checkpoint a model or part of the model.

    Checkpoint block is used to save the occupation of GPU memory in training.

    For details, please refer to `Checkpointing <https://pytorch.org/docs/stable/checkpoint.html>`_ .

    Args:
        model (torch.nn.Module): The model to be checkpointed. All kinds of modules are supported.
    
    Examples:
        >>> transformer_block = TransformerBlock(...)
        >>> checkpoint_block = CheckpointBlock(transformer_block)
        >>> y1, ... = checkpoint_block(x)
        >>> y2, ... = transformer_block(x)
        >>> assert torch.allclose(y1, y2)
    """
    def __init__(self, inner_module : torch.nn.Module):
        super().__init__()
        self._module = inner_module
        # build large parameter&grad here
        self._param_info = []
        self._storage_params : Dict[str, torch.nn.Parameter] = {}
        self._storage_info = {}
        self._ready = False
        self._optimization = copy.deepcopy(config["default_block_optimization"])
        if self.offload_parameter:
            self._on_device = False
        else:
            self._on_device = True
        self.profile = ModelProfile()
        self.profile.switch_off_()
        self._offloaded_hidden = None
        # sort parameters by name
        ordered_parameters = list(self._module.named_parameters())

        # calc total number of parameters
        for name, param in ordered_parameters:
            if not isinstance(param, DistributedParameter):
                raise ValueError("All parameters in checkpoint block must be DistributedParameter.")

            storage_type = storage_type_cuda(param.storage_type())
            kw_name = _get_param_kw(param)

            if kw_name not in self._storage_info:
                self._storage_info[kw_name] = {
                    "total": 0,
                    "storage_type": storage_type,
                    "requires_grad": param.requires_grad,
                    "group": param.group,
                    "dtype": param.dtype,
                }

            param_shape = param._original_shape

            self._storage_info[kw_name]["total"] = round_up(
                self._storage_info[kw_name]["total"] + param_shape.numel(), 
                512 // param.element_size()
                # 512 bytes aligned
            )

        offsets = {}
        # intialize storage buffers
        partitioned_param_memory = 0
        partitioned_grad_memory = 0
        gathered_param_memory = 0
        gathered_grad_memory = 0
        for kw, val in self._storage_info.items():
            val["world_size"] = config["world_size"]
            partition_size = round_up(val["total"], val["world_size"]) // val["world_size"]
            val["partition_size"] = partition_size
            val["begin"] = config['rank'] * partition_size
            val["end"] = (config['rank'] + 1) * partition_size
            offsets[kw] = 0
            partitioned_param_memory += dtype2memory(val["dtype"]) * val["partition_size"]
            gathered_param_memory += dtype2memory(val["dtype"]) * val["total"]
            if val["requires_grad"]:
                partitioned_grad_memory += dtype2memory(val["dtype"]) * val["partition_size"]
                gathered_grad_memory += dtype2memory(val["dtype"]) * val["total"]
        self.profile.set_parameter_memory(
            partitioned_param = partitioned_param_memory,
            gathered_param = gathered_param_memory,
            partitioned_grad = partitioned_grad_memory,
            gathered_grad = gathered_grad_memory,
        )
        self.init_storage_buffers(offload = self.offload_parameter)

        # initialize parameters in module
        for name, param in ordered_parameters:
            param_shape = param._original_shape
            kw_name = _get_param_kw(param)

            param_st = offsets[kw_name]
            offsets[kw_name] += param_shape.numel()
            param_end = offsets[kw_name]
            offsets[kw_name] = round_up(offsets[kw_name], 512 // param.element_size())

            self._param_info.append({
                "parameter": param,
                "name": name,
                "offset": param_st,
                "size": param_shape.numel(),
                "shape": param_shape,
                "kw_name": kw_name,
            })

            # copy values to buffer for normal parameter
            storage_st = self._storage_info[kw_name]["begin"]
            storage_end = self._storage_info[kw_name]["end"]
            
            # make parameter contiguous in storage
            with torch.no_grad():
                contiguous_param = OpAllGather.apply(param)

            if not (param_st >= storage_end or param_end <= storage_st):
                self._param_info[-1]["in_this_partition"] = True
                # copy offset in parameter storage
                offset_st = max(storage_st - param_st, 0)
                offset_end = min(storage_end - param_st, contiguous_param.numel())
                assert offset_st < offset_end

                # copy to offset in buffer storage
                to_offset_st = offset_st + param_st - storage_st
                to_offset_end = offset_end + param_st - storage_st
                
                # copy to buffer
                # PyTorch 1.11 changed the API of storage.__getitem__
                self._param_info[-1]["begin"] = to_offset_st
                self._param_info[-1]["end"] = (to_offset_end - to_offset_st,)
                d_dtype = self._storage_params[kw_name].dtype
                d_device = self._storage_params[kw_name].device
                param._set_partition(offset_st, offset_end)
                if self.offload_parameter:
                    param.data = torch.tensor([], dtype=d_dtype, device=d_device).set_(contiguous_param.storage(), offset_st, (offset_end - offset_st,))
                    param.allocate_cpu_storage()
                    param.release_gpu_storage()
                else:
                    param.data = torch.tensor([], dtype=param.dtype, device=param.device).set_(self._storage_params[kw_name].storage(), to_offset_st, (to_offset_end - to_offset_st,))
                    param.data[:] = \
                        torch.tensor([], dtype=d_dtype, device=d_device).set_(contiguous_param.storage(), offset_st, (offset_end - offset_st,))[:]

                del contiguous_param
            else:
                self._param_info[-1]["in_this_partition"] = False
                param.data = torch.tensor([], dtype=param.dtype, device=param.device)

            # clear parameter data, but keep the dtype and device
            setattr(param, "_in_checkpoint_block", True)

        for kw in offsets.keys():
            assert offsets[kw] == self._storage_info[kw]["total"]

    def init_storage_buffers(self, offload):
        for kw, val in self._storage_info.items():
            if offload:
                # Parameter does not resident on device.
                partition_size = 0
            else:
                partition_size = val["partition_size"]
            storage_type = val["storage_type"]
            storage_param_buffer = storage_type(partition_size)
            dtype = storage_param_buffer.dtype
            device = storage_param_buffer.device
            # bind storage to buffer tensor
            storage_tensor = torch.tensor([], dtype=dtype, device=device).set_(storage_param_buffer)
            storage_param = torch.nn.Parameter(storage_tensor)
            if val["requires_grad"]:
                storage_param.requires_grad_(True)
            else:
                storage_param.requires_grad_(False)
            self._storage_params[kw] = storage_param
    
    @property
    def optimization(self):
        return self._optimization
    @optimization.setter
    def optimization(self, optim):
        if self.offload_parameter:
            self.offload_parameter = optim["offload_parameter"]
            self.zero_level = optim["zero_level"]
        else:
            self.zero_level = optim["zero_level"]
            self.offload_parameter = optim["offload_parameter"]

        if self.offload_hidden_state:
            self.offload_hidden_state = optim["offload_hidden_state"]
            self.checkpointing = optim["checkpointing"]
        else:
            self.checkpointing = optim["checkpointing"]
            self.offload_hidden_state = optim["offload_hidden_state"]

        self.economical_forward = optim["economical_forward"]
        self.economical_backward = optim["economical_backward"]
        self.segment_synchronization = optim["segment_synchronization"]

    @property
    def segment_synchronization(self):
        return self._optimization["segment_synchronization"]
    @segment_synchronization.setter
    def segment_synchronization(self, value):
        value = bool(value)
        self._optimization["segment_synchronization"] = value

    @property
    def economical_forward(self):
        return self._optimization["economical_forward"]
    @economical_forward.setter
    def economical_forward(self, value):
        value = bool(value)
        self._optimization["economical_forward"] = value

    @property
    def economical_backward(self):
        return self._optimization["economical_backward"]
    @economical_backward.setter
    def economical_backward(self, value):
        value = bool(value)
        self._optimization["economical_backward"] = value

    @property
    def offload_parameter(self):
        return self._optimization["offload_parameter"]
    @offload_parameter.setter
    def offload_parameter(self, value):
        value = bool(value)
        if value == self._optimization["offload_parameter"]:
            return
        with torch.no_grad():
            ctx = CheckpointBlockContext(self)
            ctx.enter_prefetch()
            self._optimization["offload_parameter"] = value
            ctx.exit_offload()
    def offload_parameter_(self, offload : bool = True):
        self.offload = offload
        return self
    
    @property
    def offload_hidden_state(self):
        return self._optimization["offload_hidden_state"]
    @offload_hidden_state.setter
    def offload_hidden_state(self, value):
        value = bool(value)
        if value == True and self.checkpointing == False:
            raise RuntimeError("Hidden state offloading conflicts with non-checkpointing. Please set checkpointing = True first.")
        self._optimization["offload_hidden_state"] = value
    def offload_hidden_state_(self, value = True):
        slef.offload_hidden_state = value

    @property
    def checkpointing(self):
        return self._optimization["checkpointing"]
    @checkpointing.setter
    def checkpointing(self, value):
        value = bool(value)
        if value == False and self.offload_hidden_state == True:
            raise RuntimeError("Non-checkpointing conflicts with hidden state offloading. Please set offload_hidden_state = False first.")
        self._optimization["checkpointing"] = value
    def checkpointing_(self, value = True):
        self.checkpointing = value

    @property
    def zero_level(self):
        return self._optimization["zero_level"]
    @zero_level.setter
    def zero_level(self, value):
        assert value == 3 or value == 2
        if value == self._optimization["zero_level"]:
            return
        self._optimization["zero_level"] = value
    def zero_level_(self, value : int):
        self.zero_level = value
        return self

    def __call__(self, *args, **kwargs):
        # gather here
        placeholder = torch.tensor([], requires_grad=torch.is_grad_enabled())
        all_inputs = list(args)
        for kw, val in kwargs.items():
            all_inputs.append(kw)
            all_inputs.append(val)
        is_train = self.training and torch.is_grad_enabled()
        return OpCheckpointBlock.apply(placeholder, self, True, len(args), is_train, *all_inputs)
    def __getattr__(self,name:str):
        if name=="_module":
            return self._module
        return getattr(self._module, name)
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
    def __getattribute__(self, name: str):
        if name=="_parameters":
            return self._module._parameters
        return super().__getattribute__(name)
    def __delattr__(self, name):
        object.__delattr__(self, name)
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        raise RuntimeError("._save_to_state_dict() of CheckpointBlock should not be called")
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # gather here
        with torch.no_grad():
            with CheckpointBlockContext(self):
                return self._module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        all_keys = []
        for it in self._param_info:
            key = prefix + it["name"]
            all_keys.append(key)
            if key in state_dict:
                # load here
                input_param = state_dict[key]
                if input_param.shape != it["shape"]:
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, it["shape"]))
                    continue

                # not in this partition
                if not it["in_this_partition"]:
                    continue
                    
                # copy to buffer
                assert input_param.numel() == it["size"]
                contiguous_param = input_param.to(it["parameter"].dtype).cuda().contiguous()
                it["parameter"]._copy_data(contiguous_param)
                del contiguous_param
            elif strict:
                missing_keys.append(key)

        if strict:
            all_keys = set(all_keys)
            for key in state_dict.keys():
                if key.startswith(prefix) and key not in all_keys:
                    unexpected_keys.append(key)
        
    def grouped_parameters(self):
        assert not self.offload_parameter
        ret = {}
        for kw, val in self._storage_info.items():
            if val["group"] not in ret:
                ret[val["group"]] = []
            ret[val["group"]].append(self._storage_params[kw])
        for kw, val in ret.items():
            yield kw, val

    def init_parameters(self):
        """
        Initialize distributed parameters in this block.
        """
        for it in self._param_info:
            param = it["parameter"]
            if isinstance(param, DistributedParameter) and param._init_method is not None:
                # initialzie here
                tmp_tensor = torch.empty(it["shape"], device=param.device, dtype=param.dtype)
                param._init_method(tmp_tensor)
                param_st = it["offset"]
                param_end = it["offset"] + it["size"]
                kw_name = it["kw_name"]

                # not in this partition
                storage_st = self._storage_info[kw_name]["begin"]
                storage_end = self._storage_info[kw_name]["end"]
                if param_st >= storage_end:
                    continue
                if param_end <= storage_st:
                    continue
                    
                # copy to buffer
                assert tmp_tensor.is_contiguous() and it["size"] == tmp_tensor.numel()
                
                offset_st = max(storage_st - param_st, 0)
                offset_end = min(storage_end - param_st, tmp_tensor.numel())
                assert offset_st < offset_end

                to_offset_st = offset_st + param_st - storage_st
                to_offset_end = offset_end + param_st - storage_st
                
                # copy to buffer
                # PyTorch 1.11 changed the API of storage.__getitem__
                d_dtype = self._storage_params[kw_name].dtype
                d_device = self._storage_params[kw_name].device
                param._copy_data(tmp_tensor)
                del tmp_tensor
        
    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        return self._module._named_members(get_members_fn, prefix, recurse)
        
    def named_modules(self, memo = None, prefix: str = '', remove_duplicate: bool = True):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
            or not

        Yields:
            (string, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
                    print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """

        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._module._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    yield m
    def named_children(self):
        return self._module.named_children()
    
    def train(self, mode: bool = True):
        self._module.train(mode)

    def eval(self):
        self._module.eval()
    
    def __repr__(self):
        return self._module.__repr__()
        
class OpTransformerBlockList(torch.autograd.Function):
    @staticmethod
    def forward(ctx, placeholder, self : 'TransformerBlockList', save_list, hidden_state, *args):
        is_train = self.training and save_list is not None
        if is_train:
            offload_list = [save_list[i][2] for i in range(len(save_list))] # Let save_list[i][2] denotes whether offload checkpoint i.
        tensors = []
        others = []
        if is_train:
            _args, args = args, []
            for arg in _args:
                if torch.is_tensor(arg):
                    _req_grad = arg.requires_grad
                    arg = arg.detach().requires_grad_(_req_grad)
                args.append(arg)
            for arg in args:
                if torch.is_tensor(arg):
                    tensors.append(arg)
                    others.append(None)
                else:
                    tensors.append(None)
                    others.append(arg)

            ctx.nontensor_inputs = others
            ctx.self = self
            ctx.save_list = copy.deepcopy(save_list)
            ctx.offload_list = copy.deepcopy(offload_list)
            ctx.num_save_needed = save_list[-1][1]+1
            ctx.hidden_size = hidden_state.size()
            ctx.offloading_checkpoint = False
            ctx.offloading_parameter = False
            layer_inspector = []
            cuda_rng_state = []

            offload_hidden_state = []
            for i in range(len(self)):
                if offload_list[i]:
                    ctx.offloading_checkpoint = True
                    if (self._modules[str(i)]._offloaded_hidden is None) or (self._modules[str(i)]._offloaded_hidden.size() != hidden_state.size()):
                        self._modules[str(i)]._offloaded_hidden = torch.empty(hidden_state.size(), dtype=hidden_state.dtype, device="cpu", pin_memory=True)
                    offload_hidden_state.append(self._modules[str(i)]._offloaded_hidden)
                else:
                    offload_hidden_state.append(None)
                if self[i].offload_parameter:
                    ctx.offloading_parameter = True
        ctx.layers_dict=[{} for _ in range(len(self))]

        block_ctxs = []
        for i in range(len(self)):
            if self._modules[str(i)].zero_level == 2 and is_train:
                flag = 1
            else:
                flag = 0
            curr_ctx = CheckpointBlockContext(self._modules[str(i)], ctx.layers_dict[i], flag)
            block_ctxs.append(curr_ctx)


        layer_inputs = []
        layer_outputs = []
        # gather parameter on load stream
        with torch.no_grad():
            block_ctxs[0].enter_prefetch()
            block_ctxs[0].enter_gather()
        for i in range(len(self)):
            curr_ctx = block_ctxs[i]
            next_ctx = block_ctxs[i+1] if i < len(self) - 1 else None

            if self._modules[str(i)].segment_synchronization:
                wait_all_stream()

            config["prefetch_stream"].wait_stream(torch.cuda.current_stream())
            config["load_stream"].wait_stream(torch.cuda.current_stream())
            with torch.no_grad():
                if not self._modules[str(i)].economical_forward:
                    if i < len(self) - 1:
                        next_ctx.enter_prefetch()
                        next_ctx.enter_gather()

            hidden_state = hidden_state.detach().requires_grad_(True)

            if is_train and save_list[i][0] == i:
                if offload_list[i]:
                    _offload_hidden_state(hidden_state, self._modules[str(i)].profile, dst = offload_hidden_state[i])
                    layer_inputs.append(offload_hidden_state[i])
                else:
                    layer_inputs.append(hidden_state)
            elif self.return_hidden_states:
                layer_inputs.append(hidden_state)
            self._modules[str(i)].profile.input_recorder(hidden_state)

            with torch.no_grad():
                if self._modules[str(i)].economical_forward:
                    if i < len(self) - 1:
                        next_ctx.enter_prefetch()
                        next_ctx.enter_gather()
                curr_ctx.enter_build_param()

            curr_block_requires_grad = is_train and not self._modules[str(i)].checkpointing
            grad_context = torch.enable_grad() if curr_block_requires_grad else torch.no_grad()
            with grad_context:
                if is_train:
                    cuda_rng_state.append( torch.cuda.get_rng_state() )

                # call inner module directly
                with ScopedTensorInspectorContext() as inspector, self._modules[str(i)].profile.timer("forward_with_grad" if curr_block_requires_grad else "forward_no_grad"):
                    hidden_state = self._modules[str(i)]._module._call_impl(hidden_state, *args)
                for it in inspector.hidden_states:
                    debug.append("_inspect_hidden_states", it)
                if is_train:
                    layer_inspector.append(inspector.hidden_states)

                if curr_block_requires_grad:
                    layer_outputs.append(hidden_state)
                else:
                    layer_outputs.append(torch.tensor([]))

            with torch.no_grad():
                curr_ctx.exit()


        hidden_state = hidden_state.detach().requires_grad_(True)
        
        if is_train:
            ctx.layer_inspector = layer_inspector
            ctx.cuda_rng_state = cuda_rng_state
            ctx.save_for_backward(*layer_inputs, *tensors, *layer_outputs)
            ctx.num_save_args = len(tensors)

        if self.return_hidden_states:
            middle_hiddens = layer_inputs 
            for mid in middle_hiddens:
                mid.requires_grad_()
            middle_hiddens = torch.stack(middle_hiddens, dim=0)
            return hidden_state, middle_hiddens
        else:
            return hidden_state, None


    @staticmethod
    def backward(ctx, grad_hidden_state : torch.Tensor, grad_middle: List[torch.Tensor]):
        def enter_next(next_ctx, op = None):
            if next_ctx is not None:
                if op is None:
                    next_ctx.enter()
                elif op == "prefetch":
                    next_ctx.enter_prefetch()
                elif op == "gather":
                    next_ctx.enter_gather()
                else:
                    raise NotImplementedError
                    

        def exit_prev(prev_ctx, prev_grad, op = None):
            if prev_ctx is not None:
                grad_context = torch.enable_grad if prev_grad else torch.no_grad
                with grad_context():
                    if op is None:
                        prev_ctx.exit()
                    elif op == "scatter":
                        prev_ctx.exit_scatter()
                    elif op == "offload":
                        prev_ctx.exit_offload()
                    else:
                        raise NotImplementedError
        

        def prefetch_input(layer_inputs, save_list, next_st, event, profile):
            if next_st < 0:
                return
            _i = save_list[next_st][1]
            if layer_inputs[_i].is_cuda:
                return
            layer_inputs[_i] = _prefetch_hidden_state(layer_inputs[_i], profile, event = event)
            layer_inputs[_i].requires_grad_(True)


        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")
        all_inputs = []
        input_requires_grad = []
        
        layer_inputs = ctx.saved_tensors[:ctx.num_save_needed]
        layer_inputs = list(layer_inputs)
        save_args = ctx.saved_tensors[ctx.num_save_needed:ctx.num_save_needed+ctx.num_save_args]
        layer_outputs = ctx.saved_tensors[ctx.num_save_needed+ctx.num_save_args:]
        for tensor, other in zip(save_args, ctx.nontensor_inputs):
            if tensor is None:
                all_inputs.append(other)
                input_requires_grad.append(False)
            else:
                # detach for tensor inputs
                input_requires_grad.append( tensor.requires_grad )
                all_inputs.append(tensor)

        with torch.random.fork_rng(devices=[torch.cuda.current_device()], enabled=True):
            with torch.enable_grad():
                blocks_ctx = []
                prefetch_event = []
                n = len(ctx.self)
                for i in range(n):
                    if ctx.self._modules[str(i)].zero_level == 2:
                        flag = 2
                    else:
                        flag = 0
                    blocks_ctx.append(CheckpointBlockContext(ctx.self._modules[str(i)], ctx.layers_dict[i], flag))
                    prefetch_event.append(torch.cuda.Event())
                # overlap load and scatter here
                prev_ctx, prev_grad, prev_st = None, False, n
                next_st = ctx.save_list[-1][0]
                next_ctx = blocks_ctx[next_st]
                enter_next(next_ctx, op = "prefetch")
                enter_next(next_ctx, op = "gather")
                prefetch_input(layer_inputs, ctx.save_list, next_st, prefetch_event[next_st], ctx.self._modules[str(next_st)].profile if next_st >= 0 else None)
                for i in reversed(range(n)):
                    if ctx.save_list[i][0] != i:
                        with torch.no_grad():
                            st = ctx.save_list[i][0]
                            for j in range(st, i):
                                if ctx.self._modules[str(j)].segment_synchronization:
                                    wait_all_stream()
                                torch.cuda.set_rng_state(ctx.cuda_rng_state[j])

                                curr_ctx = blocks_ctx[j]
                                next_ctx = blocks_ctx[j+1]
                                config["prefetch_stream"].wait_stream(torch.cuda.current_stream())
                                exit_prev(prev_ctx, prev_grad, op = "scatter")
                                enter_next(next_ctx, op = "prefetch")
                                enter_next(next_ctx, op = "gather")
                                exit_prev(prev_ctx, prev_grad, op = "offload")

                                curr_ctx.enter_build_param()
                                prefetch_event[i].wait()
                                output = ctx.self._modules[str(j)]._module._call_impl(layer_inputs[ctx.save_list[j][1]], *all_inputs)
                                prev_ctx, prev_grad, prev_st = curr_ctx, False, j
                                layer_inputs[ctx.save_list[j+1][1]] = output
                                ctx.save_list[j+1][0] = j+1
                                ctx.offload_list[j+1] = False
                
                    torch.cuda.set_rng_state(ctx.cuda_rng_state[i])
                    
                    ipt = layer_inputs[ctx.save_list[i][1]].requires_grad_(True)
                    assert ipt.is_cuda

                    if ctx.self._modules[str(i)].segment_synchronization:
                        wait_all_stream()
                    config["prefetch_stream"].wait_stream(torch.cuda.current_stream())

                    curr_ctx = blocks_ctx[i]
                    blocks_ctx[i] = None
                    next_st = ctx.save_list[i-1][0] if i>0 else -1
                    next_ctx = blocks_ctx[next_st] if i>0 else None
                    if prev_st == next_st:
                        next_ctx, prev_ctx = None, None

                    if config["nvlink_available"]:
                        exit_prev(prev_ctx, prev_grad, op = "scatter")
                        enter_next(next_ctx, op = "prefetch")
                        if not ctx.self._modules[str(i)].economical_backward:
                            # speed mode ( maybe faster )
                            prefetch_input(layer_inputs, ctx.save_list, next_st, prefetch_event[next_st], ctx.self._modules[str(next_st)].profile if next_st >= 0 else None)
                            enter_next(next_ctx, op = "gather")
                            exit_prev(prev_ctx, prev_grad, op = "offload")
                    else:
                        exit_prev(prev_ctx, prev_grad, op = "scatter")
                        exit_prev(prev_ctx, prev_grad, op = "offload")
                        if not ctx.self._modules[str(i)].economical_backward:
                            prefetch_input(layer_inputs, ctx.save_list, next_st, prefetch_event[next_st], ctx.self._modules[str(next_st)].profile if next_st >= 0 else None)
                            enter_next(next_ctx, op = "prefetch")
                            enter_next(next_ctx, op = "gather")
                    
                    curr_ctx.enter_build_param()
                    prefetch_event[i].wait()

                    if ctx.self._modules[str(i)].checkpointing:
                        with ctx.self._modules[str(i)].profile.timer("forward_with_grad"):
                            with ScopedTensorInspectorContext() as inspector:
                                output = ctx.self._modules[str(i)]._module._call_impl(ipt, *all_inputs)

                            assert len(ctx.layer_inspector[i]) == len(inspector.hidden_states), "Backward step changed"
                            for j, it in enumerate(inspector.hidden_states):
                                assert it["name"] == ctx.layer_inspector[i][j]["name"], "Backward step changed"
                                assert it["shape"] == ctx.layer_inspector[i][j]["shape"], "Backward step changed"
                                assert it["group"] == ctx.layer_inspector[i][j]["group"], "Backward step changed"
                        
                                # change the tensor in placeholder
                                ctx.layer_inspector[i][j]["requires_grad"] = it["requires_grad"]
                                ctx.layer_inspector[i][j]["tensor"] = it["tensor"]
                    else:
                        output = layer_outputs[i]

                    with ctx.self._modules[str(i)].profile.timer("backward"):
                        torch.autograd.backward(
                            [output],
                            [grad_hidden_state]
                        )
                        assert ipt.grad is not None
                        grad_hidden_state = ipt.grad
                        ipt = None
                        layer_inputs[ctx.save_list[i][1]].grad = None
                        layer_inputs[ctx.save_list[i][1]] = None
                        if grad_middle is not None:
                            grad_hidden_state = grad_hidden_state + grad_middle[i]

                    if config["nvlink_available"]:
                        if ctx.self._modules[str(i)].economical_backward:
                            # ecomonical mode ( less gpu memory )
                            enter_next(next_ctx, op = "gather")
                            exit_prev(prev_ctx, prev_grad, op = "offload")
                            prefetch_input(layer_inputs, ctx.save_list, next_st, prefetch_event[next_st], ctx.self._modules[str(next_st)].profile if next_st >= 0 else None)
                    else:
                        if ctx.self._modules[str(i)].economical_backward:
                            prefetch_input(layer_inputs, ctx.save_list, next_st, prefetch_event[next_st], ctx.self._modules[str(next_st)].profile if next_st >= 0 else None)
                            enter_next(next_ctx, op = "prefetch")
                            enter_next(next_ctx, op = "gather")

                    prev_ctx, prev_grad, prev_st = curr_ctx, True, i

                exit_prev(prev_ctx, prev_grad)

        grads = []
        for inp, requires_grad in zip(all_inputs, input_requires_grad):
            if requires_grad:
                grads.append(inp.grad)
            else:
                grads.append(None)
        return (None, None, None, grad_hidden_state) + tuple(grads)
    
class TransformerBlockList(torch.nn.Module):
    r"""
    TransformerBlockList is a list of CheckpointBlocks.

    This is designed to reduce the communication overhead by overlapping the computation and reduce_scatter operation during backward pass.

    It is similar to `torch.nn.ModuleList` but with the difference when calling .forward() and .backward().

    Example:
        >>> module_list = [ ... ]
        >>> normal_module_list = torch.nn.ModuleList(module_list)
        >>> transformer_module_list = TransformerBlockList(module_list)
        >>> # Calling normal module list
        >>> for layer in normal_module_list:
        >>>     hidden_state = layer.forward(hidden_state, ...)
        >>> # Calling transformer module list
        >>> hidden_state = transformer_module_list(hidden_state, ...)

    """
    _modules: Dict[str, CheckpointBlock]

    def __init__(self, modules: Iterable[CheckpointBlock], sqrt=False) -> None:
        super().__init__()
        
        self._modules = {}
        for i, module in enumerate(modules):
            if not isinstance(module, CheckpointBlock):
                module = CheckpointBlock(module)
            self._modules[str(i)] = module
            self.add_module(str(i), module)

        if sqrt:
            self.sqrt = True
            length = len(self)
            num_save_needed = 0
            num_freed = 0
            save_list = [None]*length
            for i in range(length-1, -1, -1):
                if num_freed == 0 or i == 0:
                    num_save_needed += 1
                    save_list[i] = [1, -num_save_needed, False]
                    num_freed = num_save_needed
                else:
                    num_freed -= 1
                    save_list[i] = [0, -(num_save_needed - num_freed), False]
            for i in range(length-1, -1, -1):
                save_list[i][1] += num_save_needed
            for i in range(0, length):
                save_list[i][0] = i if save_list[i][0]==1 else save_list[i-1][0]

            self.save_list = save_list
        else:
            self.sqrt = False

        if config["tbl_auto_optimization"]:
            self._auto_optimization = TBLAutoOptimization(self)
        else:
            self._auto_optimization = None
            
    def __len__(self) -> int:
        return len(self._modules)
    def __iter__(self) -> Iterator[CheckpointBlock]:
        return iter(self._modules.values())
    def __getitem__(self, index: Union[int, str]) -> CheckpointBlock:
        return self._modules[str(index)]

    def forward(self, hidden_state, *args, return_hidden_states = False):
        if self._auto_optimization is not None:
            self._auto_optimization.before_step()
        self.return_hidden_states = return_hidden_states
        placeholder = torch.tensor([], requires_grad=torch.is_grad_enabled())
        if self.training and torch.is_grad_enabled():
            if self.sqrt:
                save_list = self.save_list
            else:
                save_list = [(i, i, self._modules[str(i)].offload_hidden_state) for i in range(len(self))]
        else:
            save_list = None
        last_hidden, middle_hiddens = OpTransformerBlockList.apply(placeholder, self, save_list, hidden_state, *args)
        if return_hidden_states:
            return last_hidden, middle_hiddens
        else:
            return last_hidden

    def train(self, train = True):
        super().train(train)
        if self._auto_optimization is not None:
            if train:
                self._auto_optimization.train()
            else:
                self._auto_optimization.eval()

    def is_profiling(self, i = None):
        if self._auto_optimization is None:
            return False
        return self._auto_optimization.is_profiling(i = i)
