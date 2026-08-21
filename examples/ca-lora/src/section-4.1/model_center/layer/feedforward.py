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

from .linear import Linear

import math

@torch.jit.script
def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class DenseGatedACT(bmt.DistributedModule):

    def __init__(self,
                 dim_in : int,
                 dim_ff : int,
                 activate_fn : str = "gelu",
                 dtype = torch.half,
                 int8 = False,
                 init_mean = 0.0,
                 init_std = 0.02,
                 bias = False,
                 length_scale : bool = False,
        ):
        super().__init__()

        self.w_0 = Linear(
            dim_in = dim_in,
            dim_out = dim_ff,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        self.w_1 = Linear(
            dim_in = dim_in,
            dim_out = dim_ff,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        if activate_fn == "relu":
            self.act = torch.nn.ReLU()
        elif activate_fn == "gelu":
            self.act = torch.nn.GELU()
        elif activate_fn == "gelu_new":
            self.act = gelu_new
        else:
            raise ValueError("Unsupported activation function: %s" % (activate_fn))
    
    def forward(self, x : torch.Tensor):
        """ This model inherits from bmt.DistributedModule. 
            Transform an input tensor from one feature space to another via a nonlinear operation
        
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): Tensor that will be subject to nonlinear operations.

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_ff)``) 

        """
        gate_score = self.act( self.w_0(x) )
        x = self.w_1(x)

        x = gate_score * x
        return x


class DenseACT(bmt.DistributedModule):

    def __init__(self,
                 dim_in : int,
                 dim_ff : int,
                 activate_fn : str = "gelu",
                 dtype = torch.half,
                 int8 = False,
                 init_mean = 0.0,
                 init_std = 0.02,
                 bias = False,
                 length_scale : bool = False,
        ):
        super().__init__()

        self.w = Linear(
            dim_in = dim_in,
            dim_out = dim_ff,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )
        
        if activate_fn == "relu":
            self.act = torch.nn.ReLU()
        elif activate_fn == "gelu":
            self.act = torch.nn.GELU()
        elif activate_fn == "gelu_new":
            self.act = gelu_new
        else:
            raise ValueError("Unsupported activation function: %s" % (activate_fn))

    def forward(self, x : torch.Tensor):
        """ This model inherits from bmt.DistributedModule. 
            Transform an input tensor from one feature space to another via a nonlinear operation
        
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): Tensor that will be subject to nonlinear operations.

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_ff)``) 
        """
        x = self.w(x)
        x = self.act(x)
        
        return x

class FeedForward(bmt.DistributedModule):
    r"""FeedForward module

    Args:
        dim_in (int): input dimension.
        dim_ff (int): middle dimension.
        dim_out (int, optional): output dimension. Defaults to None, which means dim_in = dim_out.
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in feed-forward layer. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in feed-forward layer. Defaults to 0.02.
        bias (bool, optional): whether to use bias term in fully-connected layers used in feed-forward module. Defaults to False.
        activate_fn (str, optional): Defaults to `gated_gelu`.
        dropout_p (int, optional): Defaults to 0.
    """

    def __init__(self,
                 dim_in : int, 
                 dim_ff : int,
                 dim_out : int = None,
                 dtype = torch.half, 
                 int8 = False,
                 init_mean = 0.0, 
                 init_std = 0.02,
                 bias = False,
                 activate_fn = "gated_gelu",
                 length_scale : bool = False,
                 dropout_p = 0,
        ):

        super().__init__()

        if activate_fn.startswith("gated_"):
            self.w_in = DenseGatedACT(
                dim_in = dim_in,
                dim_ff = dim_ff,
                activate_fn = activate_fn[6:],
                dtype = dtype,
                int8 = int8,
                init_mean = init_mean,
                init_std = init_std,
                bias = bias,
                length_scale = length_scale,
            )
        else:
            self.w_in = DenseACT(
                dim_in = dim_in,
                dim_ff = dim_ff,
                activate_fn = activate_fn,
                dtype = dtype,
                int8 = int8,
                init_mean = init_mean,
                init_std = init_std,
                bias = bias,
                length_scale = length_scale,
            )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        if dim_out is None:
            dim_out = dim_in

        self.dim_ff = dim_ff
        self.dim_out = dim_out

        self.w_out = Linear(
            dim_in = dim_ff,
            dim_out = dim_out,
            length_scale = length_scale,
            length_scale_before = True,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        self.int8 = int8
        self.length_scale = length_scale

    def forward(self, x : torch.Tensor):
        """ 
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of feed-forward module.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of feed-forward module.
        """
        x = self.w_in(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.w_out(x)

        return x
