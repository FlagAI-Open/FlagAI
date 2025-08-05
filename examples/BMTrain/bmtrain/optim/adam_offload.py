import torch
from ..global_var import config
from . import _function as F
from .. import nccl
import inspect
from ..utils import check_torch_version
from copy import deepcopy
from itertools import chain
from collections import defaultdict
from ._distributed import state_dict_gather


class AdamOffloadOptimizer(torch.optim.Optimizer):
    """
    Adam optimizer using optimizer offload.
    """

    _bmtrain_optimizer = True

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        hold_steps=0,
        record_delta=False,
    ):
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
        self.avg_delta = 0
        self.var_delta = 0
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._hold_steps = hold_steps
        self._events = {}
        self.record_delta = record_delta
        if self.record_delta:
            for group in self.param_groups:
                for p in group["params"]:
                    setattr(
                        p,
                        "_delta_info",
                        (
                            torch.tensor(
                                [0 for i in range(4)], dtype=torch.float32, device="cpu"
                            )
                        ),
                    )

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
            for p in group["params"]:
                if p.grad is not None and p.requires_grad:
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )
                    if p.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                        raise RuntimeError(
                            "Adam only supports fp32, fp16 and bf16 gradients"
                        )

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros(
                            p.size(), dtype=torch.float32, device="cpu"
                        )  # on host
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros(
                            p.size(), dtype=torch.float32, device="cpu"
                        )  # on host

                        if p.dtype == torch.float32:
                            state["_param_fp32"] = torch.empty(
                                p.size(), dtype=torch.float32, pin_memory=True
                            )  # on host
                            state["_param_fp32"].copy_(p)

                            # placeholder
                            state["_grad_fp32"] = torch.empty(
                                p.size(), dtype=torch.float32, pin_memory=True
                            )  # on host
                        else:
                            state["_param_fp32"] = torch.empty(
                                p.size(), dtype=torch.float32, device="cpu"
                            )  # on host
                            state["_param_fp32"].copy_(p)

                            # placeholder
                            state["_param_fp16"] = torch.empty(
                                p.size(), dtype=p.dtype, pin_memory=True
                            )  # on host
                            state["_grad_fp16"] = torch.empty(
                                p.size(), dtype=p.dtype, pin_memory=True
                            )  # on host

                    if p not in self._events:
                        self._events[p] = torch.cuda.Event()

                    update_params.append(
                        (
                            p,
                            state,
                            self._events[p],
                            group["betas"][0],
                            group["betas"][1],
                            group["eps"],
                            group["lr"],
                            group["weight_decay"],
                        )
                    )

        # transfer parameters to host asynchronously
        for param, state, event, _, _, _, _, _ in update_params:
            if param.dtype == torch.float32:
                state["_grad_fp32"].copy_(param.grad, non_blocking=True)
            else:
                state["_grad_fp16"].copy_(param.grad, non_blocking=True)
            torch.cuda.current_stream().record_event(event)
        sum_delta = 0
        sum_sq_delta = 0
        total_numel = 0
        for param, state, event, beta1, beta2, eps, lr, weight_decay in update_params:
            # wait for transfer to host
            event.synchronize()

            # update parameters
            if param.dtype == torch.float32:
                state["_grad_fp32"].mul_(1.0 / scale)
                if ("maximize" in group) and (group["maximize"] is True):
                    grad = -state["_grad_fp32"]
                else:
                    grad = state["_grad_fp32"]
                other_kwargs = {}
                if (
                    "maximize"
                    in inspect.signature(torch.optim._functional.adam).parameters
                ):
                    other_kwargs["maximize"] = False
                torch.optim._functional.adam(
                    [state["_param_fp32"]],
                    [grad],
                    [state["exp_avg"]],
                    [state["exp_avg_sq"]],
                    [],
                    (
                        [state["step"]]
                        if check_torch_version("1.12.0") < 0
                        else [torch.tensor(state["step"])]
                    ),
                    amsgrad=False,
                    beta1=beta1,
                    beta2=beta2,
                    lr=0.0 if state["step"] < self._hold_steps else lr,
                    weight_decay=weight_decay,
                    eps=eps,
                    **other_kwargs
                )
                # transfer parameters back to device asynchronously
                param.copy_(state["_param_fp32"], non_blocking=True)
                state["step"] += 1
            else:
                state["step"] += 1
                if ("maximize" in group) and (group["maximize"] is True):
                    grad = -state["_grad_fp16"]
                else:
                    grad = state["_grad_fp16"]
                F.adam_cpu(
                    state["_param_fp32"].view(-1),
                    state["_param_fp16"].view(-1),
                    param._delta_info if self.record_delta else None,
                    grad.view(-1),
                    state["exp_avg"].view(-1),
                    state["exp_avg_sq"].view(-1),
                    beta1,
                    beta2,
                    eps,
                    0.0 if state["step"] < self._hold_steps else lr,
                    scale,
                    weight_decay,
                    state["step"],
                )
                total_numel += state["_param_fp16"].numel()
                if self.record_delta:
                    sum_delta += param._delta_info[2].item()
                    sum_sq_delta += param._delta_info[3].item()
                # transfer parameters back to device asynchronously
                param.copy_(state["_param_fp16"], non_blocking=True)
        if self.record_delta:
            self.avg_delta = sum_delta / total_numel
            self.var_delta = sum_sq_delta / total_numel - self.avg_delta**2

        return loss

    def get_avg_delta(self) -> None:
        return self.avg_delta if self.record_delta else 0

    def get_var_delta(self) -> None:
        return self.var_delta if self.record_delta else 0

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
        saved_groups = state_dict["param_groups"]

        if len(groups) != len(saved_groups):
            raise ValueError(
                "loaded state dict has a different number of " "parameter groups"
            )
        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group "
                "that doesn't match the size of optimizer's group"
            )

        # Update the state
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain.from_iterable((g["params"] for g in saved_groups)),
                chain.from_iterable((g["params"] for g in groups)),
            )
        }

        # _param_start_end = chain.from_iterable((g["params_start_end"] for g in saved_groups))
        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        is_whole = False if "is_whole" not in state_dict else state_dict["is_whole"]
        pop_key = []
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                if is_whole and param._start_partition is not None:
                    for key in ["_param_fp32", "exp_avg_sq", "exp_avg"]:
                        if key in v:
                            v[key] = v[key][
                                param._start_partition : param._end_partition
                            ]
                elif is_whole and param._start_partition is None:
                    pop_key.append(param)

                if "_param_fp32" not in v:
                    with torch.no_grad():
                        v["_param_fp32"] = torch.empty(
                            param.size(), dtype=torch.float32, device="cpu"
                        )
                        v["_param_fp32"].copy_(param)

                for name, dtype in [
                    ("exp_avg", torch.float32),
                    ("exp_avg_sq", torch.float32),
                    ("_param_fp32", torch.float32),
                ]:
                    if name in v:
                        v[name] = v[name].to("cpu").to(dtype)

                state[param] = v
                if param.dtype == torch.float32:
                    state[param]["_param_fp32"] = state[param][
                        "_param_fp32"
                    ].pin_memory()  # on host
                    # initialize placeholders
                    state[param]["_grad_fp32"] = torch.empty(
                        param.size(), dtype=torch.float32, pin_memory=True
                    )  # on host
                else:
                    # initialize placeholders
                    state[param]["_param_fp16"] = torch.empty(
                        param.size(), dtype=param.dtype, pin_memory=True
                    )  # on host
                    state[param]["_grad_fp16"] = torch.empty(
                        param.size(), dtype=param.dtype, pin_memory=True
                    )  # on host
            else:
                state[k] = v
        for k in pop_key:
            state.pop(k)

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group["params"] = group["params"]
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": state, "param_groups": param_groups})

    def state_dict(self, gather=False) -> dict:
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
            packed = {k: v for k, v in group.items() if k != "params"}
            param_mappings.update(
                {
                    id(p): i
                    for i, p in enumerate(group["params"], start_index)
                    if id(p) not in param_mappings
                }
            )
            packed["params"] = [param_mappings[id(p)] for p in group["params"]]
            start_index += len(packed["params"])
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
        packed_state = {
            (param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): cut_states(v)
            for k, v in self.state.items()
        }
        states = {
            "state": packed_state,
            "param_groups": param_groups,
        }
        if gather:
            states = state_dict_gather(states)
            states["is_whole"] = True
        else:
            states["is_whole"] = False

        return states

    # TODO zero_grad(set_to_none=True) makes optimizer crashed, maybe the reason of grad accu
    def zero_grad(self, set_to_none: bool = False):
        super().zero_grad(set_to_none=set_to_none)
