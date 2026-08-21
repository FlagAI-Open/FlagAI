import torch
from .parameter import DistributedParameter
from .global_var import config
import itertools
from .utils import tp_split_tensor

class DistributedModule(torch.nn.Module):
    """
    DistributedModule is a subclass of torch.nn.Module that overrides the `__getattr__` method to gather distributed parameters automatically.
    
    """

    def __getattr__(self, name: str):
        ret = super().__getattr__(name)
        # gather distributed parameters if not in bmt.Block
        if isinstance(ret, DistributedParameter) and not ret._in_block: 
            return ret.gather()
        return ret
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None:
                if isinstance(param, DistributedParameter):#and not param._in_block:
                    if param._in_block:
                        destination[prefix + name] = param.tp_gather().detach() # sync operation
                    else:
                        destination[prefix + name] = param.gather_all().detach() # sync operation
                    if config['save_param_to_cpu']:
                        destination[prefix + name] = destination[prefix + name].cpu()
                else:
                    if config['save_param_to_cpu']:
                        destination[prefix + name] = param if keep_vars else param.detach().cpu()
                    else:
                        destination[prefix + name] = param if keep_vars else param.detach()

        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                tp_mode = param._tp_mode
                input_param = state_dict[key]
                if input_param.__class__.__name__ == "DistributedTensorWrapper":
                    input_param = input_param.broadcast()
                # This is used to avoid copying uninitialized parameters into
                # non-lazy modules, since they dont have the hook to do the checks
                # in such case, it will error when accessing the .shape attribute.
                is_param_lazy = torch.nn.parameter.is_lazy(param)
                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if not is_param_lazy and len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if not is_param_lazy and not isinstance(param, DistributedParameter) and input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, param.shape))
                    continue
                verify_shape = torch.Size(param._original_shape if not tp_mode else param._tp_original_shape)
                if not is_param_lazy and isinstance(param, DistributedParameter) and input_param.shape != verify_shape:
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, verify_shape))
                try:
                    with torch.no_grad():
                        if isinstance(param, DistributedParameter):
                            tp_split_dim = param._tp_split_dim
                            if tp_mode and tp_split_dim >= 0:
                                input_param = tp_split_tensor(input_param, tp_split_dim)
                            param._copy_data(input_param)
                        else:
                            param.copy_(input_param)
                except Exception as ex:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}, '
                                      'an exception occurred : {}.'
                                      .format(key, param.size(), input_param.size(), ex.args))
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)

