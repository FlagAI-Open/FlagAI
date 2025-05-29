# coding=utf-8
# Copyright 2022 The OpenBMB team.
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

import torch
import bmtrain as bmt
import torch.nn.functional as F

@torch.jit.script
def rms_layernorm(hidden : torch.Tensor, weight : torch.Tensor, eps :float):
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class LayerNorm(bmt.DistributedModule):
    r"""
    `LayerNorm <https://arxiv.org/abs/1607.06450>`_ if bias = True: :math:`y = {x-\text{E}[x]\over \text{Var}[x]+\text{eps}} * w + \text{bias}`

    `RMS LayerNorm <https://arxiv.org/abs/1910.07467>`_ if bias = False: :math:`y = {x\over \text{Var}[x]+\text{eps}} * w`

    Args:
        dim_norm (int): norm dimesion
        dtype (optional): Defaults to torch.half.
        bias (bool, optional): whether to add the :math:`\text{bias}` term. Defaults to True.
        eps (float, optional): :math:`\text{eps}` term. Defaults to 1e-5.
        init_var (float, optional): weight will be all initialized to init_var. Defaults to 1.0.
    """
    def __init__(self, dim_norm : int, 
                       dtype=torch.half, 
                       bias=True, 
                       eps : float = 1e-5,
                       init_var = 1.0
                       ):

        super().__init__()

        self.eps = eps
        self.dim_norm = dim_norm
        self.weight = bmt.DistributedParameter(
            torch.ones(dim_norm, dtype=dtype) * init_var)
        self.bias = bmt.DistributedParameter(
            torch.zeros(dim_norm, dtype=dtype)) if bias else None
    
    def forward(self, x : torch.Tensor):
        """ 
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch_size, seq_len, dim_norm)``): Input tensor that need to be normalized.

        Return:
            :obj:`torch.Tensor` of shape ``(batch_size, seq_len, dim_norm)``: The layernorm output. 

        """
        assert x.size(-1) == self.dim_norm
        
        if self.bias is not None:
            return F.layer_norm(x, (self.dim_norm,), self.weight, self.bias, self.eps)
        else:
            return rms_layernorm(x, self.weight, self.eps)
