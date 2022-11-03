# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
"""attentions."""
import os
import math
import torch

from flagai.model.layers.layer_norm_bmt import CPM3bmtLayerNorm


import bmtrain as bmt


from flagai.model.layers.linear_bmt import CPM3bmtLinear


class CPM3bmtAttention(bmt.DistributedModule):
    r"""attention module consisting procedure of Q, K, V combination and its output projection. 
    For more detail, see `Attention is All you Need <https://arxiv.org/abs/1706.03762>`_.

    Args:
        dim_in (int): input dimension.
        dim_head (int): dimension of each heads used in attention.
        num_heads (int): number of heads used in attention.
        dim_out (int, optional): output dimension. Defaults to None, which means dim_in = dim_out.
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in attetion module. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in attention module. Defaults to 0.02.
        bias (bool, optional): whether to use bias term in fully-connected layers used in attention module. Defaults to False.
        mask_value (float, optional): mask value of the masked position. Defaults to `-inf`.
        pos_bias_type (str, optional): `relative` for relative position bias, `rotary` for ratery position embedding. Defaults to `none`.
        attn_scale (bool, optional): whether to scale before softmax, i.e., :math:`\text{softmax}({Q K^T \over \sqrt{\text{dim_model}}})`. Default to False.
        dropout_p (float, optional): Defaults to 0.
    """

    def __init__(self, dim_in : int, 
                       dim_head : int,
                       num_heads : int, 
                       dim_out : int = None,
                       dtype = torch.half,
                       int8 = False, 
                       init_mean = 0.0, 
                       init_std = 0.02,
                       bias = False,
                       mask_value : float = float("-inf"),
                       pos_bias_type : str = "none",
                       length_scale : bool = False,
                       attn_scale : bool = False,
                       dropout_p : float= 0,
                       ):

        super().__init__()
        if dim_out is None:
            dim_out = dim_in

        self.project_q = CPM3bmtLinear(
            dim_in = dim_in,
            dim_out = num_heads * dim_head,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        self.project_k = CPM3bmtLinear(
            dim_in = dim_in,
            dim_out = num_heads * dim_head,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        self.project_v = CPM3bmtLinear(
            dim_in = dim_in,
            dim_out = num_heads * dim_head,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        self.attention_out = CPM3bmtLinear(
            dim_in = num_heads * dim_head,
            dim_out = dim_out,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )
    
        self.dim_in = dim_in
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_out = dim_out
        self.int8 = int8
        self.length_scale = length_scale
        self.attn_scale = attn_scale
        self.mask_value = mask_value

        if dropout_p:
            self.attention_dropout = torch.nn.Dropout(dropout_p)
        else:
            self.attention_dropout = None

        self.pos_bias_type = pos_bias_type
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, 
            query : torch.Tensor,
            key_value : torch.Tensor,
            mask : torch.Tensor,
            position_bias = None,
        ):
        """ This model inherits from bmt.DistributedModule. 

        Args:
            query (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            key_value (:obj:`torch.Tensor` of shape ``(batch, len_k, dim_model)``): Length of input sequence before padding.  
            mask (:obj:`torch.Tensor` of shape ``(batch, len_q, len_k)``): Used to avoid performing attention on padding token indices.
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, len_q, len_k)`` or ``(1, num_heads, len_k, len_q)``): Provide positional information about tensor `key_value` and `query`. 

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): The attention output.
        """

        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        h_q = self.project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = self.project_k(key_value)         # (batch, len_k, num_heads * dim_head)
        h_v = self.project_v(key_value)         # (batch, len_k, num_heads * dim_head)

        h_q = h_q.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1, 3)   # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)   # (batch, num_heads, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)   # (batch, num_heads, len_k, dim_head)

        h_q = h_q.contiguous().view(batch_size * self.num_heads, len_q, self.dim_head)      # (batch * num_heads, len_q, dim_head)
        h_k = h_k.contiguous().view(batch_size * self.num_heads, len_k, self.dim_head)      # (batch * num_heads, len_k, dim_head)
        h_v = h_v.contiguous().view(batch_size * self.num_heads, len_k, self.dim_head)      # (batch * num_heads, len_k, dim_head)

        if self.pos_bias_type == "rotary":
            h_q, h_k = position_bias(h_q, h_k)

        # (batch * num_heads, len_q, dim_head) @ (batch * num_heads, len_k, dim_head)T 
        # => (batch * num_heads, len_q, len_k)
        
        score = torch.matmul( h_q, h_k.transpose(1, 2))
        if self.attn_scale:
            score = score / math.sqrt(self.dim_head)

        # (batch, num_heads, len_q, len_k) 
        score = score.view(batch_size, self.num_heads, len_q, len_k)

        if self.pos_bias_type == "relative":
            if position_bias is not None:
                # (batch, num_heads, len_q, len_k) + (1, num_heads, len_q, len_k) 
                score = score + position_bias
        
        score = torch.masked_fill(
            score,
            mask.view(batch_size, 1, len_q, len_k)==False,
            torch.scalar_tensor(self.mask_value, device=score.device, dtype=score.dtype)
        )   # (batch, num_heads, len_q, len_k)

        score = self.softmax(score)

        # avoid nan in softmax
        score = torch.masked_fill(
            score,
            mask.view(batch_size, 1, len_q, len_k)==False,
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype)
        ).view(batch_size * self.num_heads, len_q, len_k) # (batch * num_heads, len_q, len_k)

        if self.attention_dropout is not None:
            score = self.attention_dropout(score)

         # (batch * num_heads, len_q, len_k) @ (batch * num_heads, len_k, dim_head) = (batch * num_heads, len_q, dim_head)
        score = torch.matmul(score, h_v)

        score = score.view(batch_size, self.num_heads, len_q, self.dim_head).permute(0, 2, 1, 3) # (batch, len_q, num_heads, dim_head)
        score = score.reshape(batch_size, len_q, self.num_heads * self.dim_head) # (batch, len_q, num_heads * dim_head)

        # (1#batch, dim_model, num_heads * dim_head) @ (batch, num_heads * dim_head, len_q) = (batch, dim_model, len_q)
        score = self.attention_out(score)

        return score


class CPM3bmtCrossAttention(torch.nn.Module):
    """  The whole cross-attention block. A sequence of operation. Consists of layernorm, cross-attention and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.half.
        norm_init_var (float, optional): init_var used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1.0.
        norm_bias (bool, optional): bias used in :py:class:`model_center.layer.LayerNorm`. Defaults to False.
        norm_eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        att_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.Attention`. Defaults to 0.0.
        att_init_std (float, optional): init_std used in :py:class:`model_center.layer.Attention`. Defaults to 0.02.
        att_bias (bool, optional): bias used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        att_mask_value (float, optional): mask_value used in in :py:class:`model_center.layer.Attention`. Defaults to float("-inf").
        pos_bias_type (str, optional): pos_bias_type used in :py:class:`model_center.layer.Attention`. Defaults to "none".
        post_layer_norm (bool, optional): whether to use post-layernorm. Defaults to False, which means pre-layernorm.
        attn_scale (bool, optional): attn_scale used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        dropout_p (float, optional): Defaults to 0.
    """

    def __init__(self, 
                 dim_model : int, 
                 num_heads : int, 
                 dim_head : int, 
                 dtype = torch.half,
                 int8 = False, 
                 norm_init_var : float = 1.0,
                 norm_bias : bool = False,
                 norm_eps : float = 1e-5, 
                 att_init_mean : float = 0.0, 
                 att_init_std : float = 0.02,
                 att_bias : bool = False,
                 att_mask_value : float = float("-inf"),
                 pos_bias_type : str = "none",
                 post_layer_norm : bool = False,
                 length_scale : bool = False,
                 attn_scale : bool = False,
                 dropout_p : float = 0,
                ):

        super().__init__()

        self.layernorm_before_attention = CPM3bmtLayerNorm(
            dim_norm = dim_model, 
            bias = norm_bias, 
            dtype = dtype,
            eps = norm_eps, 
            init_var = norm_init_var,
        )

        self.self_attention = CPM3bmtAttention(
            dim_in = dim_model, 
            num_heads = num_heads, 
            dim_head = dim_head,
            dim_out = dim_model, 
            dtype = dtype,
            int8 = int8, 
            init_mean = att_init_mean,
            init_std = att_init_std,
            bias = att_bias,
            mask_value = att_mask_value,
            pos_bias_type = pos_bias_type,
            length_scale = length_scale,
            attn_scale = attn_scale,
            dropout_p = dropout_p,
        )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.post_layer_norm = post_layer_norm

    def forward(self,
                hidden_states : torch.Tensor,
                key_value_states: torch.Tensor,
                attention_mask : torch.Tensor,
                position_bias  = None,
            ):
        """
        Args:
            hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Input of cross-attention block. It can be seen as query in the coming self-attention operation.
            key_value_states(:obj:`torch.Tensor` of shape ``(batch, seq_cross, dim_model)``): Used as key_value in coming self_attention operation. 
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_self, seq_cross)``): Avoid invalid areas to participate in the calculation.  
            position_bias (:obj:`torch.Tensor` of shape ``(num_heads, seq_self, seq_cross)``): Provide positional information to self-attention block.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of cross-attention block.

        """ 
        x = self.layernorm_before_attention(hidden_states)
        if self.post_layer_norm:
            hidden_states = x
        x = self.self_attention(x, key_value_states, attention_mask, position_bias)
        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = hidden_states + x
        return hidden_states


class CPM3bmtSelfAttention(torch.nn.Module):
    """  The whole cross-attention block. A sequence of operation. Consists of layernorm, self-attention and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.half.
        norm_init_var (float, optional): init_var used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1.0.
        norm_bias (bool, optional): bias used in :py:class:`model_center.layer.LayerNorm`. Defaults to False.
        norm_eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        att_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.Attention`. Defaults to 0.0.
        att_init_std (float, optional): init_std used in :py:class:`model_center.layer.Attention`. Defaults to 0.02.
        att_bias (bool, optional): bias used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        att_mask_value (float, optional): mask_value used in in :py:class:`model_center.layer.Attention`. Defaults to float("-inf").
        pos_bias_type (str, optional): pos_bias_type used in :py:class:`model_center.layer.Attention`. Defaults to "none".
        post_layer_norm (bool, optional): whether to use post-layernorm. Defaults to False, which means pre-layernorm.
        attn_scale (bool, optional): attn_scale used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        dropout_p (float, optional): Defaults to 0.
    """

    def __init__(self, 
                 dim_model : int, 
                 num_heads : int, 
                 dim_head : int, 
                 dtype = torch.half,
                 int8 = False, 
                 norm_init_var : float = 1.0,
                 norm_bias : bool = False,
                 norm_eps : float = 1e-5, 
                 att_init_mean : float = 0.0, 
                 att_init_std : float = 0.02,
                 att_bias : bool = False,
                 att_mask_value : float = float("-inf"),
                 pos_bias_type : str = "none",
                 post_layer_norm : bool = False,
                 length_scale : bool = False,
                 attn_scale : bool = False,
                 dropout_p : float = 0,
                ):

        super().__init__()

        self.layernorm_before_attention = CPM3bmtLayerNorm(
            dim_norm = dim_model, 
            bias = norm_bias, 
            dtype = dtype,
            eps = norm_eps, 
            init_var = norm_init_var,
        )

        self.self_attention = CPM3bmtAttention(
            dim_in = dim_model, 
            num_heads = num_heads, 
            dim_head = dim_head,
            dim_out = dim_model, 
            dtype = dtype,
            int8 = int8, 
            init_mean = att_init_mean,
            init_std = att_init_std,
            bias = att_bias,
            mask_value = att_mask_value,
            pos_bias_type = pos_bias_type,
            length_scale = length_scale,
            attn_scale = attn_scale,
            dropout_p = dropout_p,
        )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.post_layer_norm = post_layer_norm

    def forward(self,
                hidden_states : torch.Tensor,
                attention_mask : torch.Tensor,
                position_bias = None,
            ):
        """
        Args:
            hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Input of self-attention block. It can be the embedding of a batch of sequences.
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_self, seq_self)``): Avoid invalid areas to participate in the calculation.  
            position_bias (:obj:`torch.Tensor` of shape ``(num_heads, seq_self, seq_self)``): Provide positional information to self-attention block.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of attention block.

        """    
        x = self.layernorm_before_attention(hidden_states)
        if self.post_layer_norm:
            hidden_states = x
        x = self.self_attention(x, x, attention_mask, position_bias)
        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = hidden_states + x
        return hidden_states
