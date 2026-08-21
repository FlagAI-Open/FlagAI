import torch
from ..global_var import config
import torch.optim._functional as F
from . import _cuda as C
from .. import nccl
import inspect

from copy import deepcopy
from itertools import chain
from collections import defaultdict

class AdamOptimizer(torch.optim.Optimizer):
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

        self.load_stream = torch.cuda.Stream()
        self._hold_steps = hold_steps

    def _on_justify_scale(self, old_scale, new_scale):
        delta = new_scale / old_scale
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) > 0:
                    state['exp_avg'] *= delta
                    state['exp_avg_sq'] *= delta

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

        # update parameters
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and p.requires_grad:
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    if p.dtype not in [torch.float16, torch.float32]:
                        raise RuntimeError('Adam only supports fp32 or fp16 gradients')

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros(p.size(), dtype=p.dtype, device=p.device) # on device
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros(p.size(), dtype=torch.float32, device=p.device)   # on device

                        if p.dtype == torch.half:
                            state['_param_fp32'] = torch.empty(p.size(), dtype=torch.float32, device=p.device)   # on device
                            state['_param_fp32'].copy_(p)

                    # update the steps for each param group update
                    state['step'] += 1

                    if ('maximize' in group) and (group['maximize'] is True):
                        grad = -p.grad
                    else:
                        grad = p.grad
                        
                    if p.dtype == torch.half:
                        C.f_adam(
                            state["_param_fp32"],    # fp32
                            p,                      # fp16
                            grad,                 # fp16
                            state['exp_avg'],       # fp16: m
                            state["exp_avg_sq"],    # fp32: v
                            group['betas'][0], group['betas'][1],
                            group['eps'],
                            0.0 if state["step"] <= self._hold_steps else group['lr'],
                            scale,
                            group['weight_decay'],
                            state['step']
                        )
                    else:
                        other_kwargs = {}
                        if 'maximize' in inspect.signature(F.adam).parameters:
                            other_kwargs['maximize'] = False
                        F.adam(
                            [p],
                            [grad / scale],
                            [state['exp_avg']],
                            [state["exp_avg_sq"]],
                            [],
                            [state["step"]],
                            amsgrad=False,
                            beta1=group['betas'][0],
                            beta2=group['betas'][1],
                            lr=0.0 if state["step"] <= self._hold_steps else group['lr'],
                            weight_decay=group['weight_decay'],
                            eps=group['eps'],
                            **other_kwargs
                        )

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

                if param.dtype == torch.half and "_param_fp32" not in v:
                    v["_param_fp32"] = torch.empty(param.size(), dtype=torch.float32, device=param.device)
                    v["_param_fp32"].copy_(param)

                for name, dtype in [("exp_avg", param.dtype), ("exp_avg_sq", torch.float32), ("_param_fp32", torch.float32)]:
                    if name in v:
                        v[name] = v[name].to(param.device).to(dtype)

                state[param] = v
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})