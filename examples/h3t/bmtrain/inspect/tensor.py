from typing import Optional
import torch
from .. import debug
from .. import nccl
from ..global_var import config
from ..store import broadcast_object
import math


class InspectTensor:
    """This object is returned by `InspectTensorManager`.

    You can get the tensors recorded by `record_tensor`.

    """
    def __init__(self):
        self._summary = []
    
    def _set_summary(self, summary):
        for item in summary:
            item['prefix'] = "" if item["group"] is None else f'{item["group"]}.'
            
        self._summary = []

        kw_cnt = {}
        i = 0
        while i < len(summary):
            item = summary[i]
            if item["inside_pipe"] is not None:
                assert item["inside_pipe"]["st"]
                pipe_cnt = {}
                j = i
                while j < len(summary):
                    item = summary[j]
                    kw = f'{item["prefix"]}{item["name"]}'

                    assert item["inside_pipe"] is not None
                    stage_id = item["inside_pipe"]["stage_id"]
                    stages = item["inside_pipe"]["stages"]
                    st = item["inside_pipe"]["st"]
                    ed = item["inside_pipe"]["ed"]

                    if kw not in pipe_cnt:
                        pipe_cnt[kw] = 0
                    pipe_cnt[kw] += 1

                    j += 1
                    if ed:
                        break

                for stage in range(stages):
                    if stage_id == stage:
                        broadcast_object(pipe_cnt, config["pipe_comm"], src = stage)
                        for k in range(i, j):
                            item = summary[k]
                            kw = f'{item["prefix"]}{item["name"]}'
                            if kw not in kw_cnt:
                                kw_cnt[kw] = 0
                            item["name"] = f'{item["prefix"]}{kw_cnt[kw]}.{item["name"]}'
                            item["shape"] = ((item["shape"][0] * config['micros'],) + item["shape"][1:])
                            if item['tensor'].grad is not None:
                                grad = torch.cat([summary[k+m*(j-i)]['tensor'].grad for m in range(config['micros'])], dim=0)
                            else:
                                grad = None
                            item["tensor"] = torch.cat([summary[k+m*(j-i)]['tensor'] for m in range(config['micros'])], dim=0)
                            item["tensor"].grad = grad
                            self._summary.append(item)
                            kw_cnt[kw] += 1
                    else:
                        cnt = broadcast_object({}, config["pipe_comm"], src = stage)
                        for kw, val in cnt.items():
                            if kw not in kw_cnt:
                                kw_cnt[kw] = 0
                            for _ in range(val):
                                self._summary.append({
                                    "name": f'{item["prefix"]}{kw_cnt[kw]}.{item["name"]}',
                                    "group": None,
                                    "requires_grad": None,
                                    "min": None,
                                    "max": None,
                                    "mean": None,
                                    "std": None,
                                    "shape": None,
                                    "grad_mean" : None,
                                    "grad_std" : None,
                                    "tensor": None,
                                    "inside_pipe": {"stage_id": stage},
                                })
                                kw_cnt[kw] += 1

                i = i + config['micros'] * (j - i)
            else:
                kw = f'{item["prefix"]}{item["name"]}'
                if kw not in kw_cnt:
                    kw_cnt[kw] = 0
                item["name"] = f'{item["prefix"]}{kw_cnt[kw]}.{item["name"]}'
                self._summary.append(item)
                kw_cnt[kw] += 1
                i = i + 1


    
    def get_summary(self):
        """Get the summary of the tensors recorded by `record_tensor`.

        Returns:
            A list of dicts. Each dict contains the following keys:
                - name: The name of the tensor.
                - min: The minimum value of the tensor.
                - max: The maximum value of the tensor.
                - mean: The mean value of the tensor.
                - std: The standard deviation of the tensor.
                - shape: The shape of the tensor.
                - grad_mean: The mean value of the gradient of the tensor.
                - grad_std: The standard deviation of the gradient of the tensor.

        **Note:** This method must be called outside of the `with` block.

        """
        ret = []
        for item in self._summary:
            comm = config["zero_comm"] if item["inside_pipe"] else config["comm"]
            if item["tensor"] is not None:
                if not item["requires_grad"] or item["tensor"].grad is None:
                    x = item["tensor"]
                    info = torch.empty(2, dtype=x.dtype, device=x.device)
                    info[0] = x.mean()
                    info[1] = x.var()
                    nccl.allReduce(
                        info.storage(),
                        info.storage(),
                        "sum",
                        comm
                    ) / nccl.commCount(comm)
                    x_mean = info[0].cpu().item()
                    x_std = math.sqrt(info[1].cpu().item())
                    grad_mean = None
                    grad_std = None
                else:
                    x = item["tensor"]
                    info = torch.empty(4, dtype=x.dtype, device=x.device)
                    info[0] = x.mean()
                    info[1] = x.var()
                    info[2] = x.grad.mean()
                    info[3] = x.grad.var()
                    nccl.allReduce(
                        info.storage(),
                        info.storage(),
                        "sum",
                        comm
                    ) / nccl.commCount(comm)
                    x_mean = info[0].cpu().item()
                    x_std = math.sqrt(info[1].cpu().item())
                    grad_mean = info[2].cpu().item()
                    grad_std = math.sqrt(info[3].cpu().item())

                info[0] = x.max()
                info[1] = -x.min()
                nccl.allReduce(
                    info.storage(),
                    info.storage(),
                    'max',
                    comm
                )
                x_max = info[0].cpu().item()
                x_min = - info[1].cpu().item()

                summary = {
                    "name": item["name"],
                    "min": x_min,
                    "max": x_max,
                    "mean": x_mean,
                    "std": x_std,
                    "shape": item["shape"],
                    "grad_mean" : grad_mean,
                    "grad_std" : grad_std
                }

                if item["inside_pipe"] is not None:
                    broadcast_object(summary, config["pipe_comm"], item["inside_pipe"]["stage_id"])
            else:
                summary = broadcast_object({}, config["pipe_comm"], item["inside_pipe"]["stage_id"])

            ret.append(summary)
        return ret
    
    def get_tensor(self, name : str, group : Optional[str] = None, index : Optional[int] = None) -> torch.Tensor:
        """Get the tensor recorded by `record_tensor` by name, group and index.

        Args:
            name (str): The name of the tensor.
            group (Optional[str]): The group of the tensor.
            index (Optional[int]): The index of the tensor.
        
        Returns:
            The tensor if found, otherwise None.
        
        """
        group_name_prefix = f"{group}." if group is not None else ""

        all_names = []
        if index is None:
            all_names.append(f"{group_name_prefix}{name}")
            all_names.append(f"{group_name_prefix}0.{name}")
        else:
            all_names.append(f"{group_name_prefix}{index}.{name}")

        for item in self._summary:
            if item["name"] in all_names:
                return item["tensor"]
        return None


class InspectTensorManager:
    def __init__(self) -> None:
        self._inspector = None

    def __enter__(self) -> InspectTensor:
        self.prev_val = debug.get("_inspect_tensor", False)
        if not self.prev_val:
            debug.set("_inspect_tensor", True)
            self._inspector = InspectTensor()
            return self._inspector
        else:
            raise RuntimeError("InspectTensorManager is already in use")
    
    def __exit__(self, *args):
        if not self.prev_val:
            debug.set("_inspect_tensor", self.prev_val)
            summary = debug.get("_inspect_hidden_states", [])
            self._inspector._set_summary(summary)
            self._inspector = None
            debug.set("_inspect_hidden_states", [])
    

def inspect_tensor() -> InspectTensorManager:
    """**inspect_tensor** returns a context manager that can be used to get the intermediate results of the model computations and their gradients.

    Example:
        >>> with bmt.inspect.inspect_tensor() as inspector:
        >>>     loss = model(inputs)
        >>>     loss.backward()
        >>> summary = inspector.get_summary()
        >>> text_summary = bmt.inspect.format_summary(summary)
        >>> bmt.print_rank(text_summary)
        name   shape     max     min     std     mean    grad_std  grad_mean
        ...

    **Note:** loss.backward() must be called inside the context manager, otherwise the gradients will not be recorded.
    **Note:** Calling get_summary() has significant overhead.

    """

    return InspectTensorManager()

def record_tensor(x : torch.Tensor, name : str, group = None, requires_grad = True):
    """Record the tensor for inspection.

    Args:
        x (torch.Tensor): The tensor to be recorded.
        name (str): The name of the tensor.
        group (str): The group name of the tensor.
        requires_grad (bool): Whether records the gradients of the tensor.
    
    **Note:** This function is only available in inspect_tensor context.
    **Note:** Recording too many tensors may cause memory issues.
    
    """
    if isinstance(x, torch.nn.Parameter):
        raise RuntimeError("Cannot inspect Parameter")
    
    if not debug.get("_inspect_tensor", False):
        # do nothing
        return
    
    x_shape = tuple((x.size(0) * config["world_size"],) + x.size()[1:])

    if requires_grad and x.requires_grad:
        x.retain_grad()
    debug.append("_inspect_hidden_states", {
        "name": name,
        "group": group,
        "requires_grad": requires_grad and x.requires_grad,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "shape": x_shape,
        "grad_mean" : None,
        "grad_std" : None,
        "tensor": x,
        "inside_pipe": None,
    })
