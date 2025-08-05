import torch
from ..distributed import all_reduce, all_gather


def state_dict_gather(state_dict):
    param_key = [
        p for param_group in state_dict["param_groups"] for p in param_group["params"]
    ]
    for k, v in state_dict["state"].items():
        if "step" in v:
            step = v["step"]

    for k in param_key:
        if k not in state_dict["state"]:
            state_dict["state"][k] = {
                "exp_avg": torch.tensor([], device="cuda", dtype=torch.float32),
                "exp_avg_sq": torch.tensor([], device="cuda", dtype=torch.float32),
                "_param_fp32": torch.tensor([], device="cuda", dtype=torch.float32),
                "step": step,
            }
        v = state_dict["state"][k]
        for name, dtype in [
            ("exp_avg", torch.float32),
            ("exp_avg_sq", torch.float32),
            ("_param_fp32", torch.float32),
        ]:
            if name in v:
                with torch.no_grad():
                    numel = torch.tensor(
                        v[name].numel(), device="cuda", dtype=torch.long
                    )
                    max_numel = all_reduce(numel, op="max")
                    v_p = torch.nn.functional.pad(
                        v[name], (0, max_numel - numel), value=-1e15
                    )
                    if max_numel > 0:
                        whole_state = all_gather(v_p.cuda()).flatten()
                        whole_state = whole_state[whole_state != -1e15]
                    v[name] = whole_state.contiguous().cpu()
    return state_dict
