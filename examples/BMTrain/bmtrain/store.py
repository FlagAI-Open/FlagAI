from collections import OrderedDict
from typing import Dict
import torch

from .pipe_layer import PipelineTransformerBlockList
from .block_layer import TransformerBlockList
from .global_var import config
from .block_layer import Block
from . import nccl
import io, pickle
from typing import Mapping
import threading
import bmtrain as bmt

def _save_to_state_dict(model : torch.nn.Module, rank, destination, prefix):
    if isinstance(model, Block):
        if rank != 0:
            destination = OrderedDict() # creates an temporary ordered dict
            destination._metadata = OrderedDict()
        model.state_dict(destination=destination, prefix=prefix, keep_vars=False)
    else:
        if rank != 0:
            destination = OrderedDict() # creates an temporary ordered dict
            destination._metadata = OrderedDict()
        model._save_to_state_dict(destination, prefix, False)

def _save_to_local_rank0(model : torch.nn.Module, destination=None, prefix=''):
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(version=model._version)
    _save_to_state_dict(model, config['local_rank'], destination, prefix)
    for name, module in model._modules.items():
        if module is not None:
            _save_to_local_rank0(module, destination, prefix + name + '.')
    for hook in model._state_dict_hooks.values():
        hook_result = hook(model, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def _save_to_rank0(model : torch.nn.Module, destination=None, prefix=''):
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(version=model._version)
    if not isinstance(model, PipelineTransformerBlockList):
        _save_to_state_dict(model, config['rank'], destination, prefix)
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

def _save_to_infer_model(model : torch.nn.Module, infer_model, destination=None, prefix=''):
    config['save_param_to_cpu'] = False
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(version=model._version)
    _save_to_state_dict(model, config['local_rank'], destination, prefix)
    for name, module in model._modules.items():
        if module is not None:
            if isinstance(module, TransformerBlockList):
                for local_name, local_module in module._modules.items():
                    local_state_dict = _save_to_local_rank0(local_module, None, prefix + name + "." + local_name + '.')
                    if config['local_rank'] == 0:
                        infer_model.load_layer_state_dict(local_state_dict)
            else:
                _save_to_infer_model(module, infer_model, destination, prefix + name + '.')
    for hook in model._state_dict_hooks.values():
        hook_result = hook(model, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result

    if config['local_rank'] == 0:
        infer_model.load_layer_state_dict(destination)
        

def async_save_to_file(state_dict, file_path):
    torch.save(state_dict, file_path)
    config['finish_save'] = True
    print("finish save state_dict to ", file_path) 

def save(model : torch.nn.Module, file_name : str, non_blocking : bool=False):
    """Saves the model to the file.

    Similar to torch.save, but it used for distributed modules.

    Args:
        model (torch.nn.Module): The model to be saved.
        file_name (str): The file name of the checkpoint.
        non_blocking (bool): Whether to asynchronously save state_dict to file


    Examples:
        >>> bmtrain.save(model, "model.pt")
    """
    torch.cuda.synchronize()
    state_dict = _save_to_rank0(model)
    if config["rank"] == 0:
        if non_blocking is False:
            torch.save(state_dict, file_name)
        else:
            if 'finish_save' not in config:
                config['finish_save'] = True

            if config['finish_save'] is False:
                config['save_thread'].join()

            config['finish_save'] = False
            config['save_thread'] = threading.Thread(target=async_save_to_file, args=(state_dict, file_name))
            config['save_thread'].start()
    bmt.synchronize()

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

def allgather_objects(obj):
    if bmt.world_size() == 1:
        return [obj]

    with torch.no_grad():
        data_bytes: bytes = pickle.dumps(obj)
        data_length: int = len(data_bytes)

        gpu_data_length = torch.tensor([data_length], device="cuda", dtype=torch.long)
        gathered_length = bmt.distributed.all_gather(gpu_data_length).view(-1).cpu()
        max_data_length = gathered_length.max().item()

        gpu_data_bytes = torch.zeros(max_data_length, dtype=torch.uint8, device="cuda")
        byte_storage = torch.ByteStorage.from_buffer(data_bytes)
        gpu_data_bytes[:data_length] = torch.ByteTensor(byte_storage)

        gathered_data = bmt.distributed.all_gather(gpu_data_bytes).cpu()

        ret = []
        for i in range(gathered_data.size(0)):
            data_bytes = gathered_data[i, : gathered_length[i].item()].numpy().tobytes()
            ret.append(pickle.loads(data_bytes))
        return ret

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
class DistributedTensorWrapper:
    def __init__(self, tensor, shape=None):
        self._dtype = tensor.dtype
        self._device = tensor.device
        self.shape = shape
        self.tensor = tensor
        
    def broadcast(self):
        output_param = torch.empty(self.shape, dtype=self._dtype, device="cuda")
        if config['rank'] == 0:
            input_param = self.tensor
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
        return self.tensor
    
    def __getattribute__(self, name):
        if name == "tensor" or name == "shape":
            return object.__getattribute__(self, name)
        else:
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                pass

            return getattr(self.tensor, name)

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

        if config['rank'] != 0:
            return DistributedTensorWrapper(torch.tensor([], dtype=DTYPE_LIST[dtype_idx], device="cuda"), shape=shape_list)
        else:
            return DistributedTensorWrapper(self._state_dict[key], shape=shape_list)

        
        
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
