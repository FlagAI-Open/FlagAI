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
import math
import torch.nn.functional as F

class CPM3Linear(torch.nn.Module):
    r"""A fully connected layer, which performs :math:`\pmb{y} = \mathbf{W} \pmb{x} + \pmb{b}`

    Args:
        dim_in (int): input dimension of :math:`\pmb{x}`
        dim_out (int): output dimension of :math:`\pmb{y}`
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)`. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)`. Defaults to 1.
        bias (bool, optional): whether to add bias term :math:`\pmb{b}`. Defaults to False.
    """
    def __init__(self,
                 dim_in : int,
                 dim_out : int,
                 length_scale : bool = False,
                 length_scale_before : bool = False,
                 dtype = torch.half,
                 int8 : bool = False,
                 bias : bool = False,
                ):
        super().__init__()
        self.dim_in = dim_in
        self.weight = torch.nn.Parameter(
            torch.empty((dim_out, dim_in), dtype=dtype)
        )
        self.bias = torch.nn.Parameter(
            torch.empty((dim_out,), dtype=dtype)
        ) if bias else None
        self.length_scale = length_scale
        self.length_scale_before = length_scale_before
        self.int8 = int8

    def forward(self, x : torch.Tensor):
        """ 
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer

        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.

        """
        if self.length_scale and self.length_scale_before:
            x = x / math.sqrt(self.dim_in)
        x = F.linear(x, self.weight)
        if self.length_scale and not self.length_scale_before:
            x = x / math.sqrt(self.dim_in)
        if self.bias is not None:
            x = x + self.bias
        return x