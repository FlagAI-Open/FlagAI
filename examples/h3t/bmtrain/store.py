from collections import OrderedDict
from typing import Dict
import torch

from .pipe_layer import PipelineTransformerBlockList
from .global_var import config
from .block_layer import CheckpointBlock
from . import nccl
import io, pickle
from typing import Mapping

def _save_to_state_dict(model : torch.nn.Module, destination, prefix):
    if isinstance(model, CheckpointBlock):
        if config['rank'] != 0:
            destination = OrderedDict() # creates an temporary ordered dict
            destination._metadata = OrderedDict()
        model.state_dict(destination=destination, prefix=prefix, keep_vars=False)
    else:
        if config['rank'] != 0:
            destination = OrderedDict() # creates an temporary ordered dict
            destination._metadata = OrderedDict()
        model._save_to_state_dict(destination, prefix, False)

def _save_to_rank0(model : torch.nn.Module, destination=None, prefix=''):
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(version=model._version)
    if not isinstance(model, PipelineTransformerBlockList):
        _save_to_state_dict(model, destination, prefix)
        for name, module in model._modules.items():
            if module is not None:
                _save_to_rank0(module, destination, prefix + name + '.')
        for hook in model._state_dict_hooks.values():
            hook_result = hook(model, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
    else:
        model._save_to_state_dict(destination, prefix, False)
    return destination
        


def save(model : torch.nn.Module, file_name : str):
    """Saves the model to the file.

    Similar to torch.save, but it used for distributed modules.

    Args:
        model (torch.nn.Module): The model to be saved.
        file_name (str): The file name of the checkpoint.

    Examples:
        >>> bmtrain.save(model, "model.pt")
    """
    torch.cuda.synchronize()
    state_dict = _save_to_rank0(model)
    if config["rank"] == 0:
        torch.save(state_dict, file_name)

DTYPE_LIST = [
    torch.float64,
    torch.float32,
    torch.float16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.bfloat16,
    torch.bool
]

_pickler = pickle.Pickler
_unpickler = pickle.Unpickler
def allgather_object(obj, comm):
    f = io.BytesIO()
    _pickler(f).dump(obj)
    byte_storage = torch.ByteStorage.from_buffer(f.getvalue())
    # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
    # Otherwise, it will casue 100X slowdown.
    # See:
    byte_tensor = torch.ByteTensor(byte_storage).cuda()
    all_bytes_tensors = torch.empty(byte_tensor.numel() * nccl.commCount(comm), dtype=torch.uint8, device="cuda")
    nccl.allGather(
        byte_tensor.storage(),
        all_bytes_tensors.storage(),
        comm
    )
    obj_list = []
    for i in range(nccl.commCount(comm)):
        buf = all_bytes_tensors[i*byte_tensor.numel():(i+1)*byte_tensor.numel()].cpu().numpy().tobytes()
        obj = _unpickler(io.BytesIO(buf)).load()
        obj_list.append(obj)
    return obj_list
def broadcast_object(obj, comm, src = 0):
    if nccl.commRank(comm) == src:
        f = io.BytesIO()
        _pickler(f).dump(obj)
        byte_storage = torch.ByteStorage.from_buffer(f.getvalue())
        # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
        # Otherwise, it will casue 100X slowdown.
        # See: https://github.com/pytorch/pytorch/issues/65696
        byte_tensor = torch.ByteTensor(byte_storage).cuda()
        local_size = torch.LongTensor([byte_tensor.numel()]).cuda()

        nccl.broadcast(
            local_size.storage(),
            local_size.storage(),
            src,
            comm
        )
        nccl.broadcast(
            byte_tensor.storage(),
            byte_tensor.storage(),
            src,
            comm
        )
    else:
        local_size = torch.LongTensor([0]).cuda()
        nccl.broadcast(
            local_size.storage(),
            local_size.storage(),
            src,
            comm
        )
        byte_tensor_size = local_size[0].item()
        byte_tensor = torch.empty(int(byte_tensor_size), dtype=torch.uint8, device="cuda")
        nccl.broadcast(
            byte_tensor.storage(),
            byte_tensor.storage(),
            src,
            comm
        )
        buf = byte_tensor.cpu().numpy().tobytes()
        obj = _unpickler(io.BytesIO(buf)).load()
    return obj
    
# Must be a Mapping after pytorch 1.12.0
class DistributedStateDictWrapper(Mapping):
    def __init__(self, state_dict : Dict) -> None:
        self._state_dict = state_dict
        self._metadata = broadcast_object(getattr(state_dict, "_metadata", None), config["comm"])
    
    def __getitem__(self, key : str):
        tmp_shape = torch.zeros(32, device="cuda", dtype=torch.int32)
        if config['rank'] == 0:
            input_param : torch.Tensor = self._state_dict[key]
            shape_list = torch.tensor(list(input_param.size()), device="cuda", dtype=torch.int32)
            dtype_idx = DTYPE_LIST.index(input_param.dtype)
            
            assert dtype_idx != -1, "Unknown data type %s" % input_param.dtype

            tmp_shape[0] = shape_list.size(0)
            tmp_shape[1] = dtype_idx
            tmp_shape[2:2 + shape_list.size(0)] = shape_list

        nccl.broadcast(
            tmp_shape.storage(),
            tmp_shape.storage(),
            0,
            config['comm']
        )

        shape_list_size = tmp_shape[0].item()
        dtype_idx = tmp_shape[1].item()
        shape_list = torch.Size(tmp_shape[2: 2 + shape_list_size].tolist())

        output_param = torch.empty(shape_list, dtype=DTYPE_LIST[dtype_idx], device="cuda")
        
        if config['rank'] == 0:
            input_param : torch.Tensor = self._state_dict[key]
            if input_param.is_cuda:
                input_param = input_param.clone().contiguous()
            else:
                input_param = input_param.cuda().contiguous()

            nccl.broadcast(
                input_param.storage(),
                output_param.storage(),
                0,
                config['comm']
            )
        else:
            nccl.broadcast(
                output_param.storage(),
                output_param.storage(),
                0,
                config['comm']
            )
        
        return output_param
        
    def copy(self):
        return self

    def __len__(self):
        return broadcast_object(len(self._state_dict), config["comm"])
    
    def __contains__(self, key : str):
        return broadcast_object(key in self._state_dict, config["comm"])
    
    def keys(self):
        return broadcast_object(list(self._state_dict.keys()),config["comm"])

    def __iter__(self):
        # pytorch 1.12.0 updated the load_state_dict method, which needs the state_dict to be a `Mapping`.
        return iter(self.keys())

def load(model : torch.nn.Module, file_name : str, strict : bool = True):
    """Loads the model from the file.

    Similar to torch.load, but it uses less memory when loading large models.

    Args:
        model (torch.nn.Module): The model to be loaded.
        file_name (str): The file name of the checkpoint.
        strict (bool): Strict option of `load_state_dict`.
    
    Example:
        >>> bmtrain.load(model, "model.pt", strict=True)
    """
    if config['rank'] == 0:
        state_dict = DistributedStateDictWrapper(torch.load(file_name))
    else:
        state_dict = DistributedStateDictWrapper({})

    ret = model.load_state_dict(
        state_dict,
        strict = strict
    )
    torch.cuda.synchronize()
    return ret
