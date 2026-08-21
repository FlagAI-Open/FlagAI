import torch
from ..global_var import config
from . import _cpu as C
from . import _cuda as G
from .. import nccl
import torch.optim._functional as F
import inspect
import time

from copy import deepcopy
from itertools import chain
from collections import defaultdict

class AdamOffloadOptimizer(torch.optim.Optimizer):
    """
    Adam optimizer
    """
    _bmtrain_optimizer = True

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, hold_steps=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self._hold_steps = hold_steps

    @torch.no_grad()
    def step(self, closure=None, scale=1):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # parameters to be updated
        update_params = []

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    if not p.on_host:
                        p.allocate_cpu_storage()
                    if p.dtype not in [torch.float16, torch.float32]:
                        raise RuntimeError('Adam only supports fp32 or fp16 gradients')

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        _p, p = p, p.cpu_parameter
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros(p.size(), dtype=torch.float32, device="cpu")         # on host
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros(p.size(), dtype=torch.float32, device="cpu")      # on host
                        if p.dtype == torch.half:
                            state['_param_fp32'] = torch.empty(p.size(), dtype=torch.float32, device="cpu")     # on host
                            state['_param_fp32'].copy_(p)
                        p = _p

                    update_params.append((p, state, group['betas'][0], group['betas'][1], group['eps'], group['lr'], group['weight_decay']))

        for param, state, beta1, beta2, eps, lr, weight_decay in update_params:
            # wait for transfer to host
            param.event.synchronize()
            if param.cpu_parameter.grad is None:
                continue
            if param.cpu_parameter.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

            if param._optimizer_timer is not None:
                param._optimizer_timer.enter()

            state["step"] += 1
            if ('maximize' in group) and (group['maximize'] is True):
                grad = -param.cpu_parameter.grad
            else:
                grad = param.cpu_parameter.grad

            # update parameters
            if param.dtype == torch.half:
                C.f_adam_cpu(
                    state["_param_fp32"].view(-1),
                    param.cpu_parameter.data.view(-1),
                    grad.view(-1),
                    state["exp_avg"].view(-1),
                    state["exp_avg_sq"].view(-1),
                    beta1, beta2,
                    eps,  0.0 if state["step"] <= self._hold_steps else lr,
                    scale,
                    weight_decay,
                    state["step"]
                )
            else:
                grad.mul_(1.0 / scale)
                other_kwargs = {}
                if 'maximize' in inspect.signature(F.adam).parameters:
                    other_kwargs['maximize'] = False
                F.adam(
                    [p.cpu_parameter.data],
                    [grad],
                    [state["exp_avg"]],
                    [state["exp_avg_sq"]],
                    [],
                    [state["step"]],
                    amsgrad=False,
                    beta1=beta1,
                    beta2=beta2,
                    lr=0.0 if state["step"] <= self._hold_steps else lr,
                    weight_decay=weight_decay,
                    eps=eps,
                    **other_kwargs
                )
            
            if param._optimizer_timer is not None:
                param._optimizer_timer.exit()

            # transfer parameters back to device asynchronously
            if param.on_device:
                with torch.cuda.stream(config["prefetch_stream"]):
                    param.prefetch(allocate_gpu_storage = False, non_blocking = True)

        return loss

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain.from_iterable((g['params'] for g in saved_groups)),
                      chain.from_iterable((g['params'] for g in groups)))}

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]

                if "_param_fp32" not in v:
                    v["_param_fp32"] = torch.empty(param.size(), dtype=torch.float32, device="cpu")
                    v["_param_fp32"].copy_(param)
                    
                for name, dtype in [("exp_avg", torch.float32), ("exp_avg_sq", torch.float32), ("_param_fp32", torch.float32)]:
                    if name in v:
                        v[name] = v[name].to("cpu").to(dtype)

                state[param] = v
                if param.dtype == torch.half:
                    # initialize placeholders
                    state[param]["_param_fp16"] = torch.empty(param.size(), dtype=torch.float16, pin_memory=True)  # on host
                    state[param]["_grad_fp16"] = torch.empty(param.size(), dtype=torch.float16, pin_memory=True)   # on host
                else:
                    state[param]["_param_fp32"] = state[param]["_param_fp32"].pin_memory()

                    # initialize placeholders
                    state[param]["_grad_fp32"] = torch.empty(param.size(), dtype=torch.float32, pin_memory=True)   # on host
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def state_dict(self) -> dict:
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a list containing all parameter groups where each
            parameter group is a dict
        """
        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != 'params'}
            param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index)
                                   if id(p) not in param_mappings})
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed
        
        def cut_states(state):
            return {
                "step": state["step"],
                "exp_avg": state["exp_avg"],
                "exp_avg_sq": state["exp_avg_sq"],
                "_param_fp32": state["_param_fp32"],
            }
        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): cut_states(v)
                        for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def zero_grad(self):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        foreach = self.defaults.get('foreach', False)

        if not hasattr(self, "_zero_grad_profile_name"):
            self._hook_for_profile()
        if foreach:
            per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
            for group in self.param_groups:
                for _p in group['params']:
                    if not _p.on_host:
                        _p.allocate_cpu_storage()
                    for p in [_p, _p.cpu_parameter]:
                        if p.grad is not None:
                            if p.grad.grad_fn is not None:
                                p.grad.detach_()
                            else:
                                p.grad.requires_grad_(False)
                            if (not foreach or p.grad.is_sparse):
                                p.grad.zero_()
                            else:
                                per_device_and_dtype_grads[p.grad.device][p.grad.dtype].append(p.grad)
                    _p._grad_zeroed = True
            if foreach:
                for _, per_dtype_grads in per_device_and_dtype_grads.items():
                    for grads in per_dtype_grads.values():
                        torch._foreach_zero_(grads)


