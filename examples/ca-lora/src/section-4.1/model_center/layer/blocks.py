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

from .attention import Attention, SparseSelfAttention
from .layernorm import LayerNorm
from .feedforward import FeedForward
import bmtrain as bmt
from typing import *


class SelfAttentionBlock(torch.nn.Module):
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
                 sparse_attention : bool = False,
                 attention_window : int = 512,
        ):

        super().__init__()

        self.layernorm_before_attention = LayerNorm(
            dim_norm = dim_model, 
            bias = norm_bias, 
            dtype = dtype,
            eps = norm_eps, 
            init_var = norm_init_var,
        )
        self.sparse_attention = sparse_attention
        if not sparse_attention:
            self.self_attention = Attention(
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
        else:
            self.self_attention = SparseSelfAttention(
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
                attention_window = attention_window,
            )
        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.post_layer_norm = post_layer_norm

    def forward(self,
                hidden_states : torch.Tensor,
                attention_mask : torch.Tensor,
                position_bias : Optional[torch.Tensor] = None,
                use_cache : bool = False,
                past_key_value = None,
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
        if not self.sparse_attention:
            x = self.self_attention(x, x, attention_mask, position_bias, use_cache, past_key_value)
        else:
            #no position bias for sparse attention
            x = self.self_attention(x, attention_mask, position_bias)

        if use_cache:
            x, current_key_value = x
        else:
            current_key_value = None

        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = hidden_states + x

        if use_cache:
            return hidden_states, current_key_value
        else:
            return hidden_states


class CrossAttentionBlock(torch.nn.Module):
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
                 sparse_attention : bool = False,
                 attention_window : int = 512,
        ):

        super().__init__()

        self.layernorm_before_attention = LayerNorm(
            dim_norm = dim_model, 
            bias = norm_bias, 
            dtype = dtype,
            eps = norm_eps, 
            init_var = norm_init_var,
        )

        self.sparse_attention = sparse_attention
        if not sparse_attention:
            self.self_attention = Attention(
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
        else:
            self.self_attention = SparseSelfAttention(
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
                attention_window = attention_window,
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
                position_bias : Optional[torch.Tensor] = None,
                use_cache : bool = False,
                past_key_value = None,
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

        if not self.sparse_attention:
            x = self.self_attention(x, key_value_states, attention_mask, position_bias, use_cache, past_key_value)
        else:
            #no position bias for sparse attention
            #to do
            x = self.self_attention(x, attention_mask, position_bias)

        if use_cache:
            x, current_key_value = x
        else:
            current_key_value = None

        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = hidden_states + x

        if use_cache:
            return hidden_states, current_key_value
        else:
            return hidden_states


class FFNBlock(torch.nn.Module):
    """ The whole feed-forward block. A sequence of operation. Consists of layernorm, feed-forward and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        dtype (optional): Defaults to torch.half.
        norm_init_var (float, optional): init_var used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1.0.
        norm_bias (bool, optional): bias used in :py:class:`model_center.layer.LayerNorm`. Defaults to False.
        norm_eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        ffn_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.0.
        ffn_init_std (float, optional): init_std used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.02.
        ffn_bias (bool, optional): bias used in :py:class:`model_center.layer.FeedForward`. Defaults to False.
        ffn_activate_fn (str, optional): activate_fn used in :py:class:`model_center.layer.FeedForward`. Defaults to "gated_gelu".
        post_layer_norm (bool, optional): whether to use post-layernorm. Defaults to False, which means pre-layernorm.
        dropout_p (float, optional): Defaults to 0.
    """

    def __init__(self, 
                 dim_model : int, 
                 dim_ff : int,
                 dtype = torch.half, 
                 int8 = False,
                 norm_init_var : float = 1.0,
                 norm_bias : bool = False,
                 norm_eps : float = 1e-5, 
                 ffn_init_mean : float = 0.0, 
                 ffn_init_std : float = 0.02,
                 ffn_bias : bool = False,
                 ffn_activate_fn : str = "gated_gelu",
                 post_layer_norm : bool = False,
                 length_scale : bool = False,
                 dropout_p : float = 0,
                ):
        super().__init__()

        self.layernorm_before_ffn = LayerNorm(
            dim_norm = dim_model, 
            bias = norm_bias, 
            dtype = dtype,
            eps = norm_eps, 
            init_var = norm_init_var,
        )

        self.ffn = FeedForward(
            dim_in = dim_model, 
            dim_ff = dim_ff, 
            dim_out = dim_model, 
            dtype = dtype, 
            int8 = int8,
            init_mean = ffn_init_mean, 
            init_std = ffn_init_std,
            bias = ffn_bias,
            activate_fn = ffn_activate_fn,
            length_scale = length_scale,
        )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.post_layer_norm = post_layer_norm

    def forward(self,
                hidden_states : torch.Tensor,
               ):
        """ 
        Args:
            hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Hidden states before feed forward layer.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of feed-forward block

        """ 
        x = self.layernorm_before_ffn(hidden_states)
        if self.post_layer_norm:
            hidden_states = x
        x = self.ffn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = hidden_states + x
        return hidden_states


class TransformerBlock(torch.nn.Module):
    """ The whole transformer block. A sequence of operation. Consists of self-attention block[, cross-attention block] and feed-forward block.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        is_decoder (bool, optional): whether to use cross-attention. Defaults to False.
        dtype (optional): Defaults to torch.half.
        norm_init_var (float, optional): init_var used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1.0.
        norm_bias (bool, optional): bias used in :py:class:`model_center.layer.LayerNorm`. Defaults to False.
        norm_eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        att_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.Attention`. Defaults to 0.0.
        att_init_std (float, optional): init_std used in :py:class:`model_center.layer.Attention`. Defaults to 0.02.
        att_bias (bool, optional): bias used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        att_mask_value (float, optional): mask_value used in in :py:class:`model_center.layer.Attention`. Defaults to float("-inf").
        ffn_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.0.
        ffn_init_std (float, optional): init_std used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.02.
        ffn_bias (bool, optional): bias used in :py:class:`model_center.layer.FeedForward`. Defaults to False.
        ffn_activate_fn (str, optional): activate_fn used in :py:class:`model_center.layer.FeedForward`. Defaults to "gated_gelu".
        pos_bias_type (str, optional): pos_bias_type used in :py:class:`model_center.layer.Attention`. Defaults to "none".
        post_layer_norm (bool, optional): whether to use post-layernorm. Defaults to False, which means pre-layernorm.
        attn_scale (bool, optional): attn_scale used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        dropout_p (float, optional): Defaults to 0.
    """

    def __init__(self, 
                 dim_model : int, 
                 dim_ff : int,
                 num_heads : int,
                 dim_head : int,
                 is_decoder : bool = False,
                 dtype = torch.half, 
                 int8 = False,
                 norm_init_var : float = 1.0,
                 norm_bias : bool = False,
                 norm_eps : float = 1e-5, 
                 att_init_mean : float = 0.0, 
                 att_init_std : float = 0.02,
                 att_bias : bool = False,
                 att_mask_value : float = float("-inf"),
                 ffn_init_mean : float = 0.0, 
                 ffn_init_std : float = 0.02,
                 ffn_bias : bool = False,
                 ffn_activate_fn : str = "gated_gelu",
                 pos_bias_type : str = "none",
                 post_layer_norm : bool = False,
                 parallel_ffn : bool = False,
                 length_scale : bool = False,
                 attn_scale : bool = False,
                 dropout_p : float = 0,
                 sparse_attention : bool = False,
                 attention_window : int = 512,
                 mask_att: bool = False,
                 mask_cross: bool = False,
                 mask_ffn: bool = False,
                ):
        super().__init__()

        self.mask_att = mask_att
        self.mask_cross = mask_cross
        self.mask_ffn = mask_ffn
        self.is_decoder = is_decoder

        if not mask_att:
            self.self_att = SelfAttentionBlock(
                dim_model = dim_model, 
                num_heads = num_heads, 
                dim_head = dim_head, 
                dtype = dtype,
                int8 = int8, 
                norm_eps = norm_eps, 
                norm_init_var = norm_init_var,
                norm_bias = norm_bias,
                att_init_mean = att_init_mean, 
                att_init_std = att_init_std,
                att_bias = att_bias,
                att_mask_value = att_mask_value,
                pos_bias_type = pos_bias_type,
                post_layer_norm = post_layer_norm,
                length_scale = length_scale,
                attn_scale = attn_scale,
                dropout_p = dropout_p,
                sparse_attention = sparse_attention,
                attention_window = attention_window,
            )

        if is_decoder and not mask_cross:
            self.cross_att = CrossAttentionBlock(
                dim_model = dim_model, 
                num_heads = num_heads, 
                dim_head = dim_head, 
                dtype = dtype,
                int8 = int8, 
                norm_eps = norm_eps, 
                norm_init_var = norm_init_var,
                norm_bias = norm_bias,
                att_init_mean = att_init_mean, 
                att_init_std = att_init_std,
                att_bias = att_bias,
                att_mask_value = att_mask_value,
                pos_bias_type = pos_bias_type,
                length_scale = length_scale,
                attn_scale = attn_scale,
                dropout_p = dropout_p,
            )
        else:
            self.cross_att = None

        if not mask_ffn:
            self.ffn = FFNBlock(
                dim_model = dim_model, 
                dim_ff = dim_ff,
                dtype = dtype, 
                int8 = int8,
                norm_eps = norm_eps, 
                norm_init_var = norm_init_var,
                norm_bias = norm_bias,
                ffn_init_mean = ffn_init_mean, 
                ffn_init_std = ffn_init_std,
                ffn_bias = ffn_bias,
                ffn_activate_fn = ffn_activate_fn,
                length_scale = length_scale,
                dropout_p = dropout_p,
                post_layer_norm = post_layer_norm,
            )

        self.parallel_ffn = parallel_ffn

    def forward(self,
                self_hidden_states : torch.Tensor,
                self_attention_mask : torch.Tensor,
                self_position_bias : Optional[torch.Tensor] = None,
                cross_hidden_states = None,
                cross_attention_mask = None,
                cross_position_bias = None,
                use_cache : bool = False,
                past_key_value = None,
            ):
        """
        Args:
            self_hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            self_attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_self, seq_self)``): Avoid invalid areas to participate in the calculation of self-attention.  
            self_position_bias (:obj:`torch.Tensor` of shape ``(num_heads, seq_self, seq_self)``): Provide positional information to self-attention block.
            cross_hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_cross, dim_model)``): Input of cross-attention block. 
            cross_attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_self, seq_cross)``): Avoid invalid areas to participate in the calculation of cross-attention.  
            cross_position_bias (:obj:`torch.Tensor` of shape ``(num_heads, seq_self, seq_cross)``): Provide positional information to cross-attention block.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of transformer block.

        """
        current_key_value = None
        if not self.mask_att:
            # (batch, dim_model, seq_self)
            # add positional bias on sparse attention in the future
            hidden_states = self.self_att(self_hidden_states,
                                        attention_mask = self_attention_mask,
                                        position_bias = self_position_bias,
                                        use_cache = use_cache,
                                        past_key_value = past_key_value)
            if use_cache:
                hidden_states, current_key_value = hidden_states
        else:
            hidden_states = self_hidden_states

        if self.is_decoder and self.cross_att is not None:
            if not self.mask_cross:
                # (batch, dim_model, seq_self)
                hidden_states = self.cross_att(hidden_states = hidden_states,
                                            key_value_states = cross_hidden_states,
                                            attention_mask = cross_attention_mask,
                                            position_bias = cross_position_bias)

        if not self.mask_ffn:
            # (batch, dim_model, seq_self)
            if self.parallel_ffn:
                hidden_states_2 = self.ffn(self_hidden_states)
                hidden_states = hidden_states - self_hidden_states + hidden_states_2
            else:
                hidden_states = self.ffn(hidden_states)

        if use_cache:
            return hidden_states, current_key_value
        else:
            return hidden_states

