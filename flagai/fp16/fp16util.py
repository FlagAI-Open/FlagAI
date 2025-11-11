# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FP16 utilities for mixed precision training."""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class tofp16(nn.Module):
    """
    Utility module that implements::
        def forward(self, input):
            return input.half()
    """

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


def BN_convert_float(module):
    """
    Utility function for network_to_half().
    Retained for legacy purposes.
    """
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def network_to_half(network):
    """
    Convert model to half precision in a batchnorm-safe way.
    Retained for legacy purposes. It is recommended to use FP16Model.
    """
    return nn.Sequential(tofp16(), BN_convert_float(network.half()))


def convert_module(module, dtype):
    """
    Converts a module's immediate parameters and buffers to dtype.
    """
    for param in module.parameters(recurse=False):
        if param is not None:
            if param.data.dtype.is_floating_point:
                param.data = param.data.to(dtype=dtype)
            if param._grad is not None and param._grad.data.dtype.is_floating_point:
                param._grad.data = param._grad.data.to(dtype=dtype)

    for buf in module.buffers(recurse=False):
        if buf is not None and buf.data.dtype.is_floating_point:
            buf.data = buf.data.to(dtype=dtype)


def convert_network(network, dtype):
    """
    Converts a network's parameters and buffers to dtype.
    """
    for module in network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
            continue
        convert_module(module, dtype)
        if isinstance(module, torch.nn.RNNBase) or isinstance(module, torch.nn.modules.rnn.RNNBase):
            module.flatten_parameters()
    return network


class FP16Model(nn.Module):
    """
    Convert model to half precision in a batchnorm-safe way.
    """
    def __init__(self, network):
        super(FP16Model, self).__init__()
        self.network = convert_network(network, dtype=torch.half)

    def forward(self, *inputs):
        inputs = tuple(t.half() for t in inputs)
        return self.network(*inputs)


class FP16_Module(nn.Module):
    """
    FP16 wrapper module that converts inputs to fp16 and handles mixed precision training.
    """
    def __init__(self, module):
        super(FP16_Module, self).__init__()
        self.module = module.half()

    def forward(self, *inputs, **kwargs):
        # Convert inputs to fp16
        inputs = tuple(inp.half() if inp.is_floating_point() else inp for inp in inputs)
        
        # Forward through the module
        outputs = self.module(*inputs, **kwargs)
        
        # Convert outputs back to fp32 if needed
        if isinstance(outputs, torch.Tensor):
            return outputs.float() if outputs.is_floating_point() else outputs
        elif isinstance(outputs, (tuple, list)):
            return type(outputs)(
                out.float() if isinstance(out, torch.Tensor) and out.is_floating_point() else out
                for out in outputs
            )
        else:
            return outputs

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict)


class LossScaler:
    def __init__(self, scale=1):
        self.cur_scale = scale


class DynamicLossScaler:
    def __init__(self,
                 init_scale=2**32,
                 scale_factor=2.,
                 scale_window=1000,
                 min_scale=1,
                 delayed_shift=1,
                 consecutive_hysteresis=False):
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.delayed_shift = delayed_shift
        self.cur_hysteresis = delayed_shift
        self.consecutive_hysteresis = consecutive_hysteresis

    def update_scale(self, overflow):
        if overflow:
            if self.consecutive_hysteresis:
                self.cur_hysteresis = self.delayed_shift
            else:
                self.cur_hysteresis -= 1
            if self.cur_hysteresis <= 0:
                self.cur_scale = max(self.cur_scale / self.scale_factor, self.min_scale)
                self.cur_hysteresis = self.delayed_shift
                self.last_overflow_iter = self.cur_iter
        else:
            if (self.cur_iter - self.last_overflow_iter) >= self.scale_window:
                self.cur_scale *= self.scale_factor
                self.last_overflow_iter = self.cur_iter
        self.cur_iter += 1


class FP16_Optimizer:
    """
    FP16 optimizer wrapper that handles loss scaling for mixed precision training.
    """
    def __init__(self, optimizer, static_loss_scale=1.0, dynamic_loss_scale=False, 
                 dynamic_loss_args=None):
        self.optimizer = optimizer
        self.dynamic_loss_scale = dynamic_loss_scale
        
        if dynamic_loss_scale:
            if dynamic_loss_args is None:
                dynamic_loss_args = {}
            self.loss_scaler = DynamicLossScaler(**dynamic_loss_args)
        else:
            self.loss_scaler = LossScaler(static_loss_scale)
            
        # Create master parameters in FP32
        self.master_params = []
        self.model_params = []
        
        for param_group in self.optimizer.param_groups:
            master_param_group = []
            model_param_group = []
            
            for param in param_group['params']:
                if param.requires_grad:
                    # Create FP32 master parameter
                    master_param = param.clone().float().detach()
                    master_param.requires_grad = True
                    
                    master_param_group.append(master_param)
                    model_param_group.append(param)
                    
            self.master_params.append(master_param_group)
            self.model_params.append(model_param_group)
            
            # Replace parameter groups with master parameters
            param_group['params'] = master_param_group

    def zero_grad(self):
        """Clear gradients of all optimized parameters."""
        self.optimizer.zero_grad()

    def backward(self, loss):
        """Backward pass with loss scaling."""
        scaled_loss = loss * self.loss_scaler.cur_scale
        scaled_loss.backward()

    def step(self):
        """Optimizer step with gradient unscaling and overflow checking."""
        # Check for overflow
        has_overflow = False
        for master_group, model_group in zip(self.master_params, self.model_params):
            for master_param, model_param in zip(master_group, model_group):
                if master_param.grad is not None:
                    if DynamicLossScaler._has_inf_or_nan(master_param.grad):
                        has_overflow = True
                        break
            if has_overflow:
                break
        
        # Update loss scale
        if self.dynamic_loss_scale:
            self.loss_scaler.update_scale(has_overflow)
        
        if has_overflow:
            # Skip optimizer step on overflow
            return
        
        # Unscale gradients and step optimizer
        for master_group, model_group in zip(self.master_params, self.model_params):
            for master_param, model_param in zip(master_group, model_group):
                if master_param.grad is not None:
                    master_param.grad.div_(self.loss_scaler.cur_scale)
        
        # Step optimizer
        self.optimizer.step()
        
        # Copy master parameters back to model parameters
        for master_group, model_group in zip(self.master_params, self.model_params):
            for master_param, model_param in zip(master_group, model_group):
                model_param.data.copy_(master_param.data)

    def state_dict(self):
        """Return the state dict of the wrapped optimizer."""
        return {
            'optimizer': self.optimizer.state_dict(),
            'loss_scaler': self.loss_scaler.state_dict(),
            'dynamic_loss_scale': self.dynamic_loss_scale
        }

    def load_state_dict(self, state_dict):
        """Load the state dict into the wrapped optimizer."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.loss_scaler.load_state_dict(state_dict['loss_scaler'])
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']

    @property
    def param_groups(self):
        """Return the parameter groups of the wrapped optimizer."""
        return self.optimizer.param_groups

    def add_param_group(self, param_group):
        """Add a parameter group to the wrapped optimizer."""
        # Create FP32 master parameters for the new group
        master_params = []
        for param in param_group['params']:
            if param.requires_grad:
                master_param = param.clone().float().detach()
                master_param.requires_grad = True
                master_params.append(master_param)
        param_group['params'] = master_params
        self.optimizer.add_param_group(param_group)
        # Append to internal lists
        self.master_params.append(master_params)
        self.model_params.append([p for p in param_group['params'] if p.requires_grad])
        if overflow:
            if self.consecutive_hysteresis:
                self.cur_hysteresis = self.delayed_shift
            else:
                self.cur_hysteresis -= 1
            if self.cur_hysteresis <= 0:
                self.cur_scale = max(self.cur_scale / self.scale_factor, self.min_scale)
                self.cur_hysteresis = self.delayed_shift
                self.last_overflow_iter = self.cur_iter
        else:
            if (self.cur_iter - self.last_overflow_iter) >= self.scale_window:
                self.cur_scale *= self.scale_factor
                self.last_overflow_iter = self.cur_iter
        self.cur_iter += 1

    @staticmethod
    def _has_inf_or_nan(x):
        try:
            # if x is half, the .float() incurs a deep copy, but it's necessary if 
            # Pytorch's .sum() creates a one-element tensor of the same type as x 
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually a nan/inf exception.
            # RuntimeError could come from a different source.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True
            return False

    def state_dict(self):
        return {'cur_scale': self.cur_scale,
                'cur_iter': self.cur_iter,
                'last_overflow_iter': self.last_overflow_iter,
                'scale_factor': self.scale_factor,
                'scale_window': self.scale_window,
                'min_scale': self.min_scale,
                'delayed_shift': self.delayed_shift,
                'cur_hysteresis': self.cur_hysteresis,
                'consecutive_hysteresis': self.consecutive_hysteresis}

    def load_state_dict(self, state_dict):
        self.cur_scale = state_dict['cur_scale']
        self.cur_iter = state_dict['cur_iter']
        self.last_overflow_iter = state_dict['last_overflow_iter']
        self.scale_factor = state_dict['scale_factor']
        self.scale_window = state_dict['scale_window']
        self.min_scale = state_dict['min_scale']
        self.delayed_shift = state_dict['delayed_shift']
        self.cur_hysteresis = state_dict['cur_hysteresis']
        self.consecutive_hysteresis = state_dict['consecutive_hysteresis']