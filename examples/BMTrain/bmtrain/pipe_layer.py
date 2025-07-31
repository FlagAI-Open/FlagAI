from collections import OrderedDict
import copy
import torch
import copy
from typing import Dict, Iterable, Iterator, Tuple, Union, List
import torch

from .distributed import all_gather, broadcast, all_reduce, send_activations, recv_activations 
from .global_var import config
from . import nccl
from .zero_context import (
        ZeroContext
)
from . import debug
from .block_layer import Block, round_up, _get_param_kw, _block_wrapper

class PipePreFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_state, *args):
        hidden_state_list = all_gather(hidden_state.clone(), config["pipe_comm"])
        hidden_state_list.requires_grad_()

        batch_related = args[-1]
        batch_related_origin = [True if i in args[-1] else False for i in range(len(args[:-1]))]
        batch_related_rule = []
        args = args[:-1]

        batch_size = hidden_state.shape[0]
        num_micros = config["micros"]
        args_list = [[] for _ in range(num_micros)]
        input_requires_grad = []
        for arg in args:
            if torch.is_tensor(arg):
                arg_all = all_gather(arg, config['pipe_comm'])
                if arg.dim() == hidden_state.dim() and arg.shape[0] == batch_size:
                    batch_related_rule.append(True)
                    arg_all = arg_all.flatten(0, 1).chunk(num_micros, dim=0)
                    arg_all = [tensor.requires_grad_(arg.requires_grad) for tensor in arg_all]
                else:
                    batch_related_rule.append(False)
                    arg_all = [arg_all[0].requires_grad_(arg.requires_grad) for i in range(num_micros)]
                input_requires_grad.append(arg.requires_grad)
            else:
                batch_related_rule.append(False)
                arg_all = [arg for _ in range(num_micros)]
                input_requires_grad.append(False)
            for i in range(num_micros):
                args_list[i].append(arg_all[i])
        ctx.input_requires_grad = input_requires_grad
        ctx.args_list = args_list
        if len(batch_related) == 0:
            ctx.batch_related = batch_related_rule
        else:
            ctx.batch_related = batch_related_origin
        return hidden_state_list, args_list

    @staticmethod
    def backward(ctx, grads, arg_grads):
        grads = broadcast(grads, 0, config['pipe_comm'])
        topo = config['topology']
        arg_grads = []
        num_micros = config['micros']
        for idx,requires_grad in enumerate(ctx.input_requires_grad):
            if requires_grad:
                grad = torch.cat([ctx.args_list[m][idx].grad for m in range(num_micros)], dim=0)
                grad = all_reduce(grad, "sum", config["pipe_comm"])
                split_size = topo.stages if ctx.batch_related[idx] else num_micros
                grad = grad.chunk(split_size)
                if ctx.batch_related[idx]:
                    arg_grads.append(grad[topo.stage_id])
                else:
                    arg_grads.append(grad[0])
            else:
                arg_grads.append(None)
        arg_grads.append(None) #for append(batch_related)
        return grads.chunk(topo.stages, dim=0)[topo.stage_id], *arg_grads

class PipePostFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, last_hidden, hidden_states=None, forward_stage_ranges=None, backward_stage_ranges=None, last_hidden_shape=None, return_hidden_states=False):
        topo = config['topology']
        ctx.return_hidden_states = return_hidden_states
        last_hidden = broadcast(last_hidden, config["pipe_size"] - 1, config["pipe_comm"])
        last_hidden = last_hidden.chunk(topo.stages, dim=0)
        output = last_hidden[topo.stage_id]
        output.requires_grad_()

        if return_hidden_states:
            ctx.stage_id = topo.stage_id
            ctx.stages = topo.stages
            ctx.backward_stage_ranges = backward_stage_ranges
            middle_hiddens = []
            for stage_id in range(ctx.stages):
                if ctx.stage_id == stage_id:
                    middle_hidden = hidden_states
                else:
                    middle_shape = (forward_stage_ranges[stage_id],) + last_hidden_shape
                    middle_hidden = torch.zeros(middle_shape, device=hidden_states.device, dtype=hidden_states.dtype)
                middle_hidden = broadcast(middle_hidden, stage_id, config["pipe_comm"])
                middle_hidden = middle_hidden.chunk(ctx.stages, dim=1)
                middle_hidden = middle_hidden[ctx.stage_id].clone()
                middle_hiddens.append(middle_hidden)
            middle_hiddens = torch.cat(middle_hiddens, dim=0)
            middle_hiddens.requires_grad_()
            return output, middle_hiddens
        else:
             return output

    @staticmethod
    def backward(ctx, grads, grad_middle=None):
        grad_list = all_gather(grads, config["pipe_comm"])
        grad_list = grad_list.flatten(start_dim=0, end_dim=1)

        if ctx.return_hidden_states:
            for stage_id in range(ctx.stages):
                layer_range = ctx.backward_stage_ranges[stage_id]
                grad_middle_state = grad_middle[layer_range]
                grad_middle_state = all_gather(grad_middle_state.transpose(0,1), config["pipe_comm"])
                grad_middle_state = grad_middle_state.flatten(start_dim=0, end_dim=1).transpose(0, 1)
                if ctx.stage_id == stage_id:
                    grad_hidden_state_list = grad_middle_state
            return grad_list, grad_hidden_state_list, None, None, None, None
        else:
             return grad_list

class StagePreFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, stage_id):
        ctx.stage_id = stage_id
        ctx.is_first_stage = stage_id == 0 
        ctx.is_last_stage = stage_id == config['pipe_size'] - 1
        if not ctx.is_first_stage:
            input = recv_activations(stage_id - 1, config['pipe_comm'])
            input.requires_grad_()
            return input 
        return input
        
    @staticmethod
    def backward(ctx, grad_outputs):
        if not ctx.is_first_stage:
            send_data = grad_outputs[0] if isinstance(grad_outputs, tuple) else grad_outputs 
            current_stream = torch.cuda.current_stream()
            with torch.cuda.stream(config['pp_comm_stream']):
                config['pp_comm_stream'].wait_stream(current_stream) 
                send_data.record_stream(config['pp_comm_stream'])
                send_activations(send_data, ctx.stage_id - 1, config['pipe_comm'])
        return grad_outputs, None

class StagePostFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, outputs, stage_id):
        ctx.stage_id = stage_id
        ctx.is_first_stage = stage_id == 0 
        ctx.is_last_stage = stage_id == config['pipe_size'] - 1
        if not ctx.is_last_stage:
            send_data = outputs[0] if isinstance(outputs, tuple) else outputs
            current_stream = torch.cuda.current_stream()
            with torch.cuda.stream(config['pp_comm_stream']):
                config['pp_comm_stream'].wait_stream(current_stream) 
                send_data.record_stream(config['pp_comm_stream'])
                send_activations(send_data.detach(), stage_id + 1, config['pipe_comm'])
        return outputs
        
    @staticmethod
    def backward(ctx, grad_outputs):
        if not ctx.is_last_stage:
            pre_grad_inputs = recv_activations(ctx.stage_id + 1, config['pipe_comm'])
            return pre_grad_inputs, None
        return grad_outputs, None


class PipelineTransformerBlockList(torch.nn.Module):
    r"""
    TransformerBlockList is a list of Blocks.

    This is designed to reduce the communication overhead by overlapping the computation and reduce_scatter operation during backward pass.

    It is similar to `torch.nn.ModuleList` but with the difference when calling .forward() and .backward().

    Example:
        >>> module_list = [ ... ]
        >>> normal_module_list = torch.nn.ModuleList(module_list)
        >>> transformer_module_list = PipelineTransformerBlockList(module_list)
        >>> # Calling normal module list
        >>> for layer in normal_module_list:
        >>>     hidden_state = layer.forward(hidden_state, ...)
        >>> # Calling transformer module list
        >>> hidden_state = transformer_module_list(hidden_state, ...)

    """
    _modules: Dict[str, Block]

    def __init__(self, modules: Iterable[torch.nn.Module], num_hidden=1) -> None:
        super().__init__()
        self.num_hidden = num_hidden 
        self._modules = {}
        self.layer_ids = []
        topo = config["topology"]
        self.stages = topo.stages
        self.stage_id = topo.stage_id
        self.pipe_idx = topo.pipe_idx 
        module_dict = {}
        for idx, module in enumerate(modules):
            module = _block_wrapper(module, module_dict, "PIPE")
            module._zero_level = 2 #currently, only support ZeRO-2 in pipeline mode
            self._modules[str(idx)] = module

        self.layer_ids = self.get_range_by_stage_id(self.stage_id)

        pre_module = None
        for i,layer_id in enumerate(self.layer_ids):
            module = self._modules[str(layer_id)]
            module.set_pre_module(pre_module)
            pre_module = module
            module._is_first_layer = False
            module._is_last_layer = False
            
        self._modules[str(self.layer_ids[0])]._is_first_layer = True
        self._modules[str(self.layer_ids[-1])]._is_last_layer = True
            
    def __len__(self) -> int:
        return len(self._modules) 

    def __iter__(self) -> Iterator[Block]:
        return iter(self._modules.values())

    def __getitem__(self, index: Union[int, str]) -> Block:
        return self._modules[str(index)]

    def forward(self, hidden_state, *args, batch_related=[], return_hidden_states=False):
        self.return_hidden_states = return_hidden_states
        batch_size = hidden_state.shape[0]
        num_micros = config["micros"]
        args = args + (batch_related, )
        hidden_state.requires_grad_()
        hidden_state_list, args_list = PipePreFunction.apply(hidden_state, *args)

        hidden_state_list = hidden_state_list.flatten(0, 1).chunk(num_micros, dim=0)
        outputs = []
        hidden_states = []

        for micro_idx, (hidden_state, arg) in enumerate(zip(hidden_state_list, args_list)):
            micro_hidden_states = []

            hidden_state = StagePreFunction.apply(hidden_state, self.stage_id)

            for idx,layer_id in enumerate(self.layer_ids):
                self._modules[str(layer_id)]._micro_idx = micro_idx
                if return_hidden_states:
                    micro_hidden_states.append(hidden_state)
                hidden_state = self._modules[str(layer_id)](hidden_state, *arg)
            hidden_state = StagePostFunction.apply(hidden_state, self.stage_id)

            outputs.append(hidden_state)
            if return_hidden_states:
                hidden_states.append(torch.stack(micro_hidden_states, dim=0))

        last_hidden = torch.cat(outputs, dim=0)
        last_hidden_shape = last_hidden.shape

        if return_hidden_states:
            hidden_states = torch.cat(hidden_states, dim=1) 
            forward_stage_ranges = []
            backward_stage_ranges = []
            for stage_id in range(self.stages):
                forward_stage_ranges.append(self.get_part_len_by_stage_id(stage_id))
                backward_stage_ranges.append(self.get_range_by_stage_id(stage_id))
            outputs, hidden_states = PipePostFunction.apply(last_hidden, hidden_states, forward_stage_ranges, backward_stage_ranges, last_hidden_shape, return_hidden_states)
            return outputs, hidden_states 
        else:
            outputs = PipePostFunction.apply(last_hidden)
            return outputs

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

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        for name, module in self._modules.items():
            idx = int(name)
            name = prefix + name + '.'
            
            dst = OrderedDict() # creates an temporary ordered dict
            dst._metadata = OrderedDict()

            if idx in self.layer_ids:
                with torch.no_grad():
                    with ZeroContext(module, pipe=True):
                        module._module.state_dict(destination=dst, prefix=name, keep_vars=False)

                if config["topology"].pp_zero_id == 0:
                    if config["rank"] == 0:
                        destination.update(dst)
                    else:
                        assert list(dst.keys()) == [name+n for n, parameter in module._module.named_parameters()]
                        for key, tensor in dst.items():
                            send_activations(tensor.cuda(), 0, config['pipe_comm'])
            if config['rank'] == 0 and idx not in self.layer_ids:
                for n, parameter in module._module.named_parameters():
                    destination[name+n] = recv_activations(self.get_stage_by_layer_id(idx), config['pipe_comm']).cpu()

