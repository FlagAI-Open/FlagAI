from collections import OrderedDict
import copy
import torch
import copy
from typing import Dict, Iterable, Iterator, Tuple, Union, List
import torch

from .distributed import all_gather, broadcast, all_reduce, send_activations, recv_activations
from .global_var import config
from . import nccl
from .checkpointing import ScopedTensorInspectorContext
from . import debug
from .block_layer import CheckpointBlockContext, CheckpointBlock, round_up, _get_param_kw

class OpMicroForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, placeholder, self : 'PipelineTransformerBlockList', micro_idx, block_ctx_list, layers_dict, save_list, hidden_state, *args):
        with PipeContext(self, hidden_state) as pipe_input:
            hidden_state = pipe_input[0].detach()
            tensors = [arg if torch.is_tensor(arg) else None for arg in args]
            others = [arg if not torch.is_tensor(arg) else None for arg in args]
            ctx.nontensor_inputs = others
            ctx.self = self
            ctx.micro_idx = micro_idx
            ctx.block_ctx_list = block_ctx_list
            ctx.layers_dict = layers_dict
            ctx.save_list = copy.deepcopy(save_list)
            ctx.num_save_needed = save_list[-1][1]+1
            layer_inputs = []
            layer_inspector = []
            cuda_rng_state = []
            with torch.no_grad():
                for idx,layer_id in enumerate(self.layer_ids):
                    if save_list[idx][0] == idx:
                        layer_inputs.append(hidden_state.detach())
                    cuda_rng_state.append( torch.cuda.get_rng_state() )
                    # gather parameter on load stream
                    if ctx.micro_idx == 0:
                        block_ctx_list[idx] = CheckpointBlockContext(self._modules[str(layer_id)], ctx.layers_dict[idx], 1, pipe=True)
                        block_ctx_list[idx].enter()
                    # call inner module directly
                    with ScopedTensorInspectorContext() as inspector:
                        hidden_state = self._modules[str(layer_id)]._module._call_impl(hidden_state, *args)
                    for ith, it in enumerate(inspector.hidden_states):
                        it["shape"] = ((it["shape"][0] // config['pipe_size'],) + it["shape"][1:])
                        it["inside_pipe"] = {
                            "stage_id": self.stage_id,
                            "stages": self.stages,
                            "st": (layer_id==self.layer_ids[0] and ith==0),
                            "ed": (layer_id==self.layer_ids[-1] and ith==len(inspector.hidden_states)-1),
                        }
                        debug.append("_inspect_hidden_states", it)
                    layer_inspector.append(inspector.hidden_states)
                    if ctx.micro_idx == config["micros"]-1:
                        block_ctx_list[idx].exit()
            
            ctx.layer_inspector = layer_inspector
            ctx.cuda_rng_state = cuda_rng_state

            ctx.save_for_backward(*layer_inputs, *tensors)
            pipe_input[0] = hidden_state
        if self.return_hidden_states:
            middle_hiddens = layer_inputs 
            for mid in middle_hiddens:
                mid.requires_grad_()
            middle_hiddens = torch.stack(middle_hiddens, dim=0)
            return pipe_input[0], middle_hiddens
        else:
            return pipe_input[0], None

    @staticmethod
    def backward(ctx, grad_hidden_state : torch.Tensor, grad_middle : torch.Tensor):
        def exit_prev(prev_ctx, prev_grad):
            if prev_ctx is not None:
                if prev_grad:
                    with torch.enable_grad():
                        prev_ctx.exit()
                        config["load_stream"].record_event(config["load_event"])
                else:
                    with torch.no_grad():
                        prev_ctx.exit()
                        config["load_stream"].record_event(config["load_event"])
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")
        all_inputs = []
        input_requires_grad = []
        
        layer_inputs = ctx.saved_tensors[:ctx.num_save_needed]
        save_args = ctx.saved_tensors[ctx.num_save_needed:]
        for tensor, other in zip(save_args, ctx.nontensor_inputs):
            if tensor is None:
                all_inputs.append(other)
                input_requires_grad.append(False)
            else:
                # detach for tensor inputs
                input_requires_grad.append( tensor.requires_grad )
                nw_tensor = tensor.detach()
                nw_tensor.requires_grad = tensor.requires_grad
                all_inputs.append(nw_tensor)
        with PipeContext(ctx.self, grad_hidden_state, backward=True) as pipe_input:
            grad_hidden_state = pipe_input[0]
            with torch.random.fork_rng(devices=[torch.cuda.current_device()], enabled=True):
                with torch.enable_grad():
                    # overlap load and scatter here
                    prev_ctx = None
                    prev_grad = False
                    for idx,layer_id in list(enumerate(ctx.self.layer_ids))[::-1]:
                        torch.cuda.set_rng_state(ctx.cuda_rng_state[idx])
                        ipt = layer_inputs[ctx.save_list[idx][1]].requires_grad_()
                        if ctx.micro_idx == 0:
                            ctx.block_ctx_list[idx] = CheckpointBlockContext(ctx.self._modules[str(layer_id)], ctx.layers_dict[idx], 2, pipe=True)
                            ctx.block_ctx_list[idx].enter()
                        if ctx.micro_idx == config["micros"]-1:
                            exit_prev(prev_ctx, prev_grad)
                            prev_ctx = ctx.block_ctx_list[idx]
                            prev_grad = True

                        with ScopedTensorInspectorContext() as inspector:
                            output = ctx.self._modules[str(layer_id)]._module._call_impl(ipt, *all_inputs)

                        assert len(ctx.layer_inspector[idx]) == len(inspector.hidden_states), "Backward step changed"
                        for j, it in enumerate(inspector.hidden_states):
                            it["shape"] = ((it["shape"][0] // config['pipe_size'],) + it["shape"][1:])
                            assert it["name"] == ctx.layer_inspector[idx][j]["name"], "Backward step changed"
                            assert it["shape"] == ctx.layer_inspector[idx][j]["shape"], "Backward step changed"
                            assert it["group"] == ctx.layer_inspector[idx][j]["group"], "Backward step changed"
                            
                            # change the tensor in placeholder
                            ctx.layer_inspector[idx][j]["requires_grad"] = it["requires_grad"]
                            ctx.layer_inspector[idx][j]["tensor"] = it["tensor"]
                        torch.autograd.backward(
                            [output],
                            [grad_hidden_state]
                        )
                        grad_hidden_state = ipt.grad
                        if grad_middle is not None:
                            grad_hidden_state = grad_hidden_state + grad_middle[idx]
                    if ctx.micro_idx == config["micros"]-1:
                        exit_prev(prev_ctx, prev_grad)

            pipe_input[0] = grad_hidden_state
        grads = []
        for inp, requires_grad in zip(all_inputs, input_requires_grad):
            if requires_grad:
                grads.append(inp.grad)
            else:
                grads.append(None)
        return (None, None, None, None, None, None, pipe_input[0]) + tuple(grads)

class OpPipeTransformerBlockList(torch.autograd.Function):
    @staticmethod
    def forward(ctx, placeholder, self : 'PipelineTransformerBlockList', save_list, hidden_state, *args):
        num_micros = config["micros"]
        ctx.self = self
        ctx.num_micros = num_micros
        block_ctx = [None for _ in range(len(self))]
        layers_dict = [{} for _ in range(len(self))]
        args_list = [[] for _ in range(num_micros)]
        batch_related = args[-1]
        batch_related_origin = [True if i in args[-1] else False for i in range(len(args[:-1]))]
        batch_related_rule = []
        args = args[:-1]
        batch_size = hidden_state.shape[0]
        assert (batch_size * config["pipe_size"]) % num_micros == 0, f'The batch size {(batch_size * config["pipe_size"])} must be divisible by the number of micro_batch {num_micros}'
        input_requires_grad = []
        with torch.enable_grad():
            for arg in args:
                if torch.is_tensor(arg):
                    arg_all = all_gather(arg, config['pipe_comm'])
                    if arg.shape[0] == batch_size:
                        batch_related_rule.append(True)
                        arg_all = arg_all.flatten(0, 1).chunk(num_micros, dim=0)
                        arg_all = [tensor.detach().requires_grad_(arg.requires_grad) for tensor in arg_all]
                    else:
                        batch_related_rule.append(False)
                        # assert num_micros % self.stages == 0, "batch unrelated only support num_micros % stages == 0"
                        # arg_all = [arg_all[i // (num_micros // self.stages)].detach().requires_grad_(arg.requires_grad) for i in range(num_micros)]
                        arg_all = [arg_all[0].detach().requires_grad_(arg.requires_grad) for i in range(num_micros)]
                    input_requires_grad.append(arg.requires_grad)
                else:
                    batch_related_rule.append(False)
                    arg_all = [arg for _ in range(num_micros)]
                    input_requires_grad.append(False)
                for i in range(num_micros):
                    args_list[i].append(arg_all[i])
            outputs = []
            if self.return_hidden_states:
                middles = []
            hidden_state_list = all_gather(hidden_state, config["pipe_comm"]).flatten(0, 1).detach().requires_grad_()
            ctx.hidden_state_list = hidden_state_list
            hidden_state_list = hidden_state_list.chunk(num_micros, dim=0)
            for micro_idx, (hidden_state, arg) in enumerate(zip(hidden_state_list, args_list)):
                placeholder = torch.tensor([], requires_grad=torch.is_grad_enabled())
                output, middle = OpMicroForward.apply(placeholder, self, micro_idx, block_ctx, layers_dict, save_list, hidden_state, *arg)
                outputs.append(output)
                if self.return_hidden_states:
                    middles.append(middle)
        if len(batch_related) == 0:
            ctx.batch_related = batch_related_rule
        else:
            ctx.batch_related = batch_related_origin
        ctx.args_list = args_list
        ctx.input_requires_grad = input_requires_grad
        ctx.output_list = outputs
        if self.return_hidden_states:
            ctx.middle_list = middles

        with torch.enable_grad():
            last_hidden = torch.cat(outputs, dim=0)
            last_hidden_shape = last_hidden.shape
            last_hidden = broadcast(last_hidden, config["pipe_size"] - 1, config["pipe_comm"])
            last_hidden = last_hidden.chunk(self.stages, dim=0)
            last_hidden = last_hidden[self.stage_id].clone()

        if self.return_hidden_states:
            middle_hiddens = []
            with torch.enable_grad():
                for stage_id in range(self.stages):
                    if self.stage_id == stage_id:
                        middle_hidden = torch.cat(middles, dim=1) # [(layers, micro_batch, ...), ] -> (layers, full_batch, ...)
                    else:
                        middle_shape = (self.get_part_len_by_stage_id(stage_id),)+last_hidden_shape
                        middle_hidden = torch.zeros(middle_shape, device=last_hidden.device, dtype=last_hidden.dtype)
                    middle_hidden = broadcast(middle_hidden, stage_id, config["pipe_comm"])
                    middle_hidden = middle_hidden.chunk(self.stages, dim=1)
                    middle_hidden = middle_hidden[self.stage_id].clone()
                    middle_hiddens.append(middle_hidden)
                middle_hiddens = torch.cat(middle_hiddens, dim=0)
            return last_hidden, middle_hiddens
        else:
            return last_hidden, None


    @staticmethod
    def backward(ctx, grad_hidden_state : torch.Tensor, grad_middle : torch.Tensor):
        ipt = ctx.hidden_state_list
        args_list = ctx.args_list
        input_requires_grad = ctx.input_requires_grad
        grad_hidden_state_list = all_gather(grad_hidden_state, config["pipe_comm"]).flatten(start_dim=0, end_dim=1).chunk(ctx.num_micros, dim=0)
        if ctx.self.return_hidden_states:
            for stage_id in range(ctx.self.stages):
                layer_range = ctx.self.get_range_by_stage_id(stage_id)
                grad_middle_state = grad_middle[layer_range]
                grad_middle_state = all_gather(grad_middle_state.transpose(0,1), config["pipe_comm"]).flatten(start_dim=0, end_dim=1).transpose(0, 1).chunk(ctx.num_micros, dim=1) # (layer, micro_batch, ...)
                if ctx.self.stage_id == stage_id:
                    grad_middle_state_list = grad_middle_state

            for m in range(ctx.num_micros):
                output = ctx.output_list[m]
                middle = ctx.middle_list[m]
                grad_hidden_state = grad_hidden_state_list[m]
                grad_middle_state = grad_middle_state_list[m]
                torch.autograd.backward(
                    [output, middle],
                    [grad_hidden_state, grad_middle_state],
                )
        else:
            for m in range(ctx.num_micros):
                output = ctx.output_list[m]
                grad_hidden_state = grad_hidden_state_list[m]
                torch.autograd.backward(
                    [output],
                    [grad_hidden_state],
                )
        grads = []
        for idx,requires_grad in enumerate(input_requires_grad):
            if requires_grad:
                grad = torch.cat([args_list[m][idx].grad for m in range(ctx.num_micros)], dim=0)
                grad = all_reduce(grad, "sum", config["pipe_comm"])
                split_size = ctx.self.stages if ctx.batch_related[idx] else ctx.num_micros
                grad = grad.chunk(split_size)
                if ctx.batch_related[idx]:
                    grads.append(grad[ctx.self.stage_id])
                else:
                    grads.append(grad[0])
            else:
                grads.append(None)
        grad = broadcast(ipt.grad, 0, config["pipe_comm"]).chunk(ctx.self.stages)
        grad = grad[ctx.self.stage_id]

        return (None, None, None, grad) + tuple(grads) + (None,)

class PipelineTransformerBlockList(torch.nn.Module):
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

    def __init__(self, modules: Iterable[CheckpointBlock]) -> None:
        super().__init__()
        
        self._modules = {}
        rank = config['rank']
        topo = config['topology']
        self.layer_ids = []
        pipe_group = topo.pp_group
        self.stages = topo.stages
        self.stage_id = topo.stage_id
        self.pipe_idx = topo.pipe_idx 
        for idx, module in enumerate(modules):
            if not isinstance(module, CheckpointBlock):
                module = CheckpointBlock(module)
            self._modules[str(idx)] = module

        self.layer_ids = self.get_range_by_stage_id(self.stage_id)
        self.partition_modules(self.layer_ids)
        self.next_rank = pipe_group[self.pipe_idx, self.stage_id + 1] if self.stage_id < config['pipe_size'] - 1 else -1
        self.prev_rank = pipe_group[self.pipe_idx, self.stage_id - 1] if self.stage_id > 0 else -1
        # self.micro_batches = config['num_micro_batches']
            
        self.save_list = [(i, i) for i in range(len(self.layer_ids))]
            
    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[CheckpointBlock]:
        return iter(self._modules.values())

    def __getitem__(self, index: Union[int, str]) -> CheckpointBlock:
        return self._modules[str(index)]

    def forward(self, hidden_state, *args, batch_related=[], return_hidden_states=False):
        self.return_hidden_states = return_hidden_states
        placeholder = torch.tensor([], requires_grad=torch.is_grad_enabled())
        args = list(args)
        args.append(batch_related)
        hidden_state, middle_states = OpPipeTransformerBlockList.apply(placeholder, self, self.save_list, hidden_state, *args)
        if return_hidden_states:
            return hidden_state, middle_states
        else:
            return hidden_state

    def get_range_by_stage_id(self, stage_id : int) -> List[int]:
        part_lens = [0]+[self.get_part_len_by_stage_id(i) for i in range(stage_id+1)]
        start = sum(part_lens[:stage_id+1])
        end = start + part_lens[stage_id+1]
        return range(start, end)

    def get_part_len_by_stage_id(self, stage_id : int) -> int:
        return len(self) // self.stages + (stage_id < (len(self) % self.stages))

    def get_stage_by_layer_id(self, layer_id : int) -> int:
        part_len = len(self) // self.stages
        rest = len(self) % self.stages
        if layer_id // (part_len + 1) < rest:
            return layer_id // (part_len + 1)
        else:
            return rest + (layer_id - rest * (part_len+1)) // part_len

    def partition_modules(self, idxs) -> None:
        for i in range(len(self)):
            contiguous_params = {}
            for kw, val in self[i]._storage_info.items():
                storage_type = val["storage_type"]
                contiguous_params[kw] = storage_type(round_up(val["total"], config["world_size"] // config["pipe_size"]))
                nccl.allGather(
                    self[i]._storage_params[kw].storage(),
                    contiguous_params[kw],
                    config["comm"]
                )

            if i not in idxs:
                for name, param in self[i]._module.named_parameters():
                    param.data = torch.tensor([], dtype = param.dtype, device = param.device)
                for kw, val in self[i]._storage_info.items():
                    val["begin"] = self.stage_id
                    val["end"] = self.stage_id + 1
                    val["partition_size"] = 1
                    val["total"] = val["world_size"]
                    dtype = self[i]._storage_params[kw].dtype
                    device = self[i]._storage_params[kw].device
                    self[i]._storage_params[kw] = \
                        torch.nn.Parameter(torch.tensor([0], dtype = dtype, device=device))
            else:
                for kw, val in self[i]._storage_info.items():
                    storage_type = val["storage_type"]
                    val["world_size"] = config["world_size"] // config["pipe_size"]
                    partition_size = round_up(val["total"], val["world_size"]) // val["world_size"]
                    val["partition_size"] = partition_size
                    val["begin"] = config['zero_rank'] * partition_size
                    val["end"] = (config['zero_rank'] + 1) * partition_size
                    storage_param_buffer = storage_type(partition_size)
                    dtype = storage_param_buffer.dtype
                    device = storage_param_buffer.device
                    self[i]._storage_params[kw] = torch.nn.Parameter(
                        torch.tensor([], dtype=dtype, device=device).set_(storage_param_buffer)
                    )
                    if val["requires_grad"]:
                        self[i]._storage_params[kw].requires_grad_(True)
                    else:
                        self[i]._storage_params[kw].requires_grad_(False)
                ordered_parameters = list(self[i]._module.named_parameters())
                for idx, named_param in enumerate(ordered_parameters):
                    name, param = named_param
                    param_info = self[i]._param_info[idx]
                    kw_name = _get_param_kw(param)
                    storage_info = self[i]._storage_info[kw_name]
                    storage_st = storage_info["begin"]
                    storage_end = storage_info["end"]
                    param_st = param_info["offset"]
                    param_end = param_st + param_info["size"]
                    if not (param_st >= storage_end or param_end <= storage_st):
                        # copy offset in parameter storage
                        offset_st = max(storage_st - param_st, 0)
                        offset_end = min(storage_end - param_st, param_info["size"])
                        assert offset_st < offset_end
                        to_offset_st = offset_st + param_st - storage_st
                        to_offset_end = offset_end + param_st - storage_st
                        d_dtype = self[i]._storage_params[kw_name].dtype
                        d_device = self[i]._storage_params[kw_name].device
                        param.data = torch.tensor([], dtype=param.dtype, device=param.device).set_(self[i]._storage_params[kw_name].storage(), to_offset_st, (to_offset_end - to_offset_st,))
                        param_info["begin"] = to_offset_st
                        param_info["end"] = (to_offset_end - to_offset_st,)
                        param.data[:] = \
                            torch.tensor([], dtype=d_dtype, device=d_device).set_(contiguous_params[kw], storage_st+to_offset_st, (to_offset_end - to_offset_st,))[:]
                    else:
                        param.data = torch.tensor([], dtype=param.dtype, device=param.device)
            del contiguous_params
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        for name, module in self._modules.items():
            idx = int(name)
            name = prefix + name + '.'
            
            dst = OrderedDict() # creates an temporary ordered dict
            dst._metadata = OrderedDict()

            if idx in self.layer_ids:
                with torch.no_grad():
                    with CheckpointBlockContext(module, pipe=True):
                        module._module.state_dict(destination=dst, prefix=name, keep_vars=False)
                if config["zero_rank"] == 0:
                    if config["rank"] == 0:
                        destination.update(dst)
                    else:
                        assert list(dst.keys()) == [name+n for n, parameter in module._module.named_parameters()]
                        for key, tensor in dst.items():
                            send_activations(tensor.cuda(), 0, config['pipe_comm'])
            if config['rank'] == 0 and idx not in self.layer_ids:
                for n, parameter in module._module.named_parameters():
                    destination[name+n] = recv_activations(self.get_stage_by_layer_id(idx), config['pipe_comm'])

class PipeContext:
    def __init__(self, module, hidden_state, backward=False):
        self.module = module
        self.stage_id = module.stage_id
        self.stages = module.stages
        self.next_rank = module.next_rank
        self.prev_rank = module.prev_rank
        self.hidden_state = [hidden_state]
        self.backward = backward
        self.send_buffer = {}
    def enter(self):
        if self.backward:
            if self.stage_id != self.stages -1:
                self.hidden_state[0] = recv_activations(self.stage_id + 1, config['pipe_comm'])
        else:
            if self.stage_id != 0:
                self.hidden_state[0] = recv_activations(self.stage_id - 1, config['pipe_comm'])
        return self.hidden_state
    def exit(self):
        if self.backward:
            if self.stage_id != 0:
                send_activations(self.hidden_state[0], self.stage_id - 1, config['pipe_comm'])
        else:
            if self.stage_id != self.stages - 1:
                send_activations(self.hidden_state[0], self.stage_id + 1, config['pipe_comm'])
    def __enter__(self):
        return self.enter()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()