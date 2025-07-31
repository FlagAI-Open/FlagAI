import torch
from typing import Callable, TypeVar
from functools import wraps
from . import debug

class ScopedDebugTensorList:
    def __init__(self) -> None:
        self._hidden_states = []
    
    @property
    def hidden_states(self):
        return self._hidden_states

    def _set_hidden_states(self, hidden_states):
        self._hidden_states = hidden_states

class ScopedTensorInspectorContext:
    def __init__(self):
        pass
    
    def __enter__(self):
        self.prev_hidden = debug.get("_inspect_hidden_states", [])
        debug.set("_inspect_hidden_states", [])
        self._local_list = ScopedDebugTensorList()
        return self._local_list
    
    def __exit__(self, *args):
        self._local_list._set_hidden_states(debug.get("_inspect_hidden_states", []))
        debug.set("_inspect_hidden_states", self.prev_hidden)
        self.prev_hidden = None

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, placeholder, func, preserve_rng_state, *args):
        ctx.func = func
        ctx.preserve_rng_state = preserve_rng_state
        
        ctx.cuda_rng_state = torch.cuda.get_rng_state() if preserve_rng_state else None
        
        tensors = []
        others = []
        for arg in args:
            if torch.is_tensor(arg):
                tensors.append(arg)
                others.append(None)
            else:
                tensors.append(None)
                others.append(arg)
        ctx.nontensor_inputs = others
        ctx.save_for_backward(*tensors)

        with torch.no_grad(), ScopedTensorInspectorContext() as inspector:
            outputs = func(*args)
        
        # append scoped hidden states to global list as a placeholder
        for it in inspector.hidden_states:
            debug.append("_inspect_hidden_states", it)
        ctx.inspect_list = inspector.hidden_states

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")

        all_inputs = []
        input_reqires_grad = []
        for tensor, other in zip(ctx.saved_tensors, ctx.nontensor_inputs):
            if tensor is None:
                all_inputs.append(other)
                input_reqires_grad.append(False)
            else:
                input_reqires_grad.append( tensor.requires_grad )
                nw_tensor = tensor.detach()
                nw_tensor.requires_grad = tensor.requires_grad
                all_inputs.append(nw_tensor)

        
        with torch.random.fork_rng(devices=[torch.cuda.current_device()], enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.cuda.set_rng_state(ctx.cuda_rng_state)
            with torch.enable_grad(), ScopedTensorInspectorContext() as inspector:
                outputs = ctx.func(*all_inputs)
            
            assert len(ctx.inspect_list) == len(inspector.hidden_states), "Backward step changed"
            for i, it in enumerate(inspector.hidden_states):
                assert it["name"] == ctx.inspect_list[i]["name"], "Backward step changed"
                assert it["shape"] == ctx.inspect_list[i]["shape"], "Backward step changed"
                assert it["group"] == ctx.inspect_list[i]["group"], "Backward step changed"
                
                # change the tensor in placeholder
                ctx.inspect_list[i]["tensor"] = it["tensor"]
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        
        assert len(outputs) == len(grad_outputs)

        outputs_with_grad = []
        grad_of_output = []
        for i, output in enumerate(outputs):
            if torch.is_tensor(output) and output.requires_grad:
                outputs_with_grad.append(output)
                grad_of_output.append(grad_outputs[i])
        
        torch.autograd.backward(
            outputs_with_grad,
            grad_of_output,
        )
        grads = []
        for inp, requires_grad in zip(all_inputs, input_reqires_grad):
            if requires_grad:
                grads.append(inp.grad)
            else:
                grads.append(None)
        return (None, None, None) + tuple(grads)


R = TypeVar("R")
def checkpoint(func : Callable[..., R]) -> Callable[..., R]:
    @wraps(func)
    def wrapper(*args):
        placeholder = torch.tensor([], requires_grad=torch.is_grad_enabled())
        return CheckpointFunction.apply(placeholder, func, True, *args)
    return wrapper
