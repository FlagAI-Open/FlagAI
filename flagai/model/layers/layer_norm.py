# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# layer norm

import torch
import torch.nn as nn

import torch.nn.functional as F


def rms_layernorm(hidden : torch.Tensor, weight : torch.Tensor, eps :float):
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight



class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        """Perform layer normalization to input x, with two learnable variables gamma and beta"""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        hidden_states = self.gamma * (x - mean) / (std + self.eps)

        return hidden_states + self.beta


class T5LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1,
                                                               keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states


class BertLayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class CPM3LayerNorm(torch.nn.Module):
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
        self.weight = torch.nn.Parameter(
            torch.ones(dim_norm, dtype=dtype) * init_var)
        self.bias = torch.nn.Parameter(
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