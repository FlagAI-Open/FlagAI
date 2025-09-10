import torch
from .global_var import config
from .zero_context import ZeroContext


def zero_pre_forward(module, inputs):
    """Helper function for using ZeroContext to gather parmas before forward."""
    enter = True
    pipe = False
    if module._mode == "PIPE":
        enter = module._micro_idx == 0
        pipe = True
    if enter:
        zero_level = module._zero_level
        forward_flag = 1 if zero_level == 2 else 0
        if zero_level == 2 and not module._need_release:
            forward_flag = 2  # repeating forward in same layer
        if module.all_param_no_grad:  # only forward
            forward_flag = 0
        module._forward_block_ctx = ZeroContext(module, module._layer_dict, pipe=pipe)
        module._forward_block_ctx.enter(forward_flag)


def zero_post_forward(module, inputs, outputs):
    """Helper function for module _forwar_block_ctx weather exits after forward."""
    forward_flag = 1 if module._zero_level == 2 else 0
    if module.all_param_no_grad:
        forward_flag = 0
    exit = True
    if module._mode == "PIPE":
        exit = module._micro_idx == config["micros"] - 1

    if exit:
        module._forward_block_ctx.exit(forward_flag)


def zero_pre_backward(module, grad_outputs):
    """Helper function for using ZeroContext to init grad buffer before backward."""
    backward_flag = 2 if module._zero_level == 2 else 0
    if module._mode != "PIPE":
        module._backward_block_ctx = ZeroContext(module, module._layer_dict)
        module._backward_block_ctx.enter(backward_flag, True)
        module.release_next_module(backward_flag)
    else:
        if module._micro_idx == config["micros"] - 1:
            module._backward_block_ctx = ZeroContext(
                module, module._layer_dict, pipe=True
            )
            module._backward_block_ctx.enter(backward_flag, True)


def zero_post_backward(module, grad_inputs, grad_outputs):
    """Helper function for module weather release after backward."""
    backward_flag = 2 if module._zero_level == 2 else 0
    if module._mode != "PIPE":
        if module._is_first_layer:
            module.release(backward_flag)
    else:
        if module._micro_idx == 0:
            module.release(backward_flag)
        module._micro_idx -= 1


class OneStepNoGradFunc(torch.autograd.Function):
    """
    Requires_grad = False for all inputs.
    """

    @staticmethod
    def forward(ctx, module, placeholder, *x):
        ctx.x = x
        ctx.module = module
        ctx.rng_state = torch.cuda.get_rng_state()

        with torch.no_grad():
            out = module._module(*x)
        zero_post_forward(module, None, out)
        if not isinstance(out, torch.Tensor):
            return tuple(out)
        return out

    @staticmethod
    def backward(ctx, grads):
        zero_pre_backward(ctx.module, grads)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()], enabled=True):
            torch.cuda.set_rng_state(ctx.rng_state)
            x = ctx.x
            with torch.enable_grad():
                out = ctx.module._module(*x)
                torch.autograd.backward(out, grads)
        zero_post_backward(ctx.module, grads, None)
        grads = []
        for _ in x:
            grads.append(None)
        return None, None, *grads


class PreHookFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, *x):
        ctx.module = module
        zero_pre_forward(module, x)
        return x

    @staticmethod
    def backward(ctx, *grads):
        zero_post_backward(ctx.module, grads, None)
        return None, *grads


class PostHookFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, *out):
        ctx.module = module
        zero_post_forward(module, None, out)
        return out

    @staticmethod
    def backward(ctx, *grads):
        zero_pre_backward(ctx.module, grads)
        return None, *grads
