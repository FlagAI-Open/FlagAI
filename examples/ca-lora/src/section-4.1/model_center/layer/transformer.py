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
from typing import Optional, List, Tuple

from .blocks import TransformerBlock
from .layernorm import LayerNorm

class Encoder(torch.nn.Module):
    """ Layers of encoder transformer blocks plus an final layernorm.

    Args:
        num_layers (int): number of layers.
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
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
            num_layers : int,
            dim_model : int, 
            dim_ff : int,
            num_heads : int,
            dim_head : int,
            dtype : torch.dtype = torch.half,
            int8 : bool = False, 
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
            length_scale : bool = False,
            attn_scale : bool = False,
            dropout_p : float = 0,
            parallel_ffn : bool = False,
            sparse_attention : bool = False,
            attention_window : int = 512,
            mask_modules : Optional[List[Tuple[int, int]]] = None,
        ):

        super().__init__()
        
        self.num_layers = num_layers

        if mask_modules is not None:
            assert len(mask_modules) == num_layers, "The total number of masks should equal to num_layers"
            for mask_module in mask_modules:
                assert len(mask_module) == 2, "For encoder, each mask should be (mask_att, mask_ffn)"
        else:
            mask_modules = [(False, False)] * num_layers

        self.layers = bmt.TransformerBlockList([
                bmt.CheckpointBlock(
                    TransformerBlock(
                        dim_model = dim_model, 
                        dim_ff = dim_ff,
                        num_heads = num_heads,
                        dim_head = dim_head,
                        is_decoder = False,
                        dtype = dtype, 
                        int8 = int8,
                        norm_eps = norm_eps, 
                        norm_init_var = norm_init_var,
                        norm_bias = norm_bias,
                        att_init_mean = att_init_mean, 
                        att_init_std = att_init_std,
                        att_bias = att_bias,
                        att_mask_value = att_mask_value,
                        ffn_init_mean = ffn_init_mean, 
                        ffn_init_std = ffn_init_std,
                        ffn_bias = ffn_bias,
                        ffn_activate_fn = ffn_activate_fn,
                        pos_bias_type = pos_bias_type,
                        post_layer_norm = post_layer_norm,
                        length_scale = length_scale,
                        attn_scale = attn_scale,
                        dropout_p = dropout_p,
                        parallel_ffn = parallel_ffn,
                        sparse_attention = sparse_attention,
                        attention_window = attention_window,
                        mask_att = mask_modules[ith][0],
                        mask_ffn = mask_modules[ith][1],
                    )
                )
                for ith in range(num_layers)
        ])

        self.output_layernorm = LayerNorm(
                    dim_norm = dim_model, 
                    bias = norm_bias, 
                    dtype = dtype,
                    eps = norm_eps,
                    init_var = norm_init_var)

    def forward(self, hidden_states : torch.Tensor,
                      attention_mask : torch.Tensor,
                      position_bias : torch.Tensor = None,
                      use_cache : bool = False,
                      past_key_values = None,
        ):
        """
        Args:
            hidden-states (:obj:`torch.Tensor` of shape ``(batch, seq_enc, dim_model)``): Input of encoder, might be the embedding of a batch of sequences. 
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_enc, seq_enc)``): Avoid invalid areas to participate in the calculation 
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, seq_enc, seq_enc)``) Provides position information to attention mechanism.  

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_enc, dim_model)``: The encoder output. 

        """
        if not use_cache:
            hidden_states, mid = self.layers(hidden_states, attention_mask, position_bias, None, None, None, return_hidden_states=True)
            hidden_states = self.output_layernorm(hidden_states)
            return hidden_states, mid
        else:
            with torch.no_grad():
                current_key_values = []
                for i, module in enumerate(self.layers):
                    hidden_states  = module(hidden_states, attention_mask, position_bias, 
                                            None, None, None, 
                                            past_key_value = past_key_values[i] if past_key_values else None, 
                                            use_cache = use_cache)
                    current_key_values.append(hidden_states[1])
                    hidden_states = hidden_states[0]
                hidden_states = self.output_layernorm(hidden_states)
                return hidden_states, current_key_values

class Decoder(torch.nn.Module):
    """ Layers of decoder transformer blocks plus an final layernorm.

    Args:
        num_layers (int): number of layers.
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
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
            num_layers : int,
            dim_model : int, 
            dim_ff : int,
            num_heads : int,
            dim_head : int,
            dtype : torch.dtype = torch.half,
            int8 : bool = False, 
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
            length_scale : bool = False,
            attn_scale : bool = False,
            dropout_p : float = 0,
            parallel_ffn : bool = False,
            mask_modules : Optional[List[Tuple[int, int, int]]] = None,
        ):

        super().__init__()
        
        self.num_layers = num_layers

        if mask_modules is not None:
            assert len(mask_modules) == num_layers, "The total number of masks should equal to num_layers"
            for mask_module in mask_modules:
                assert len(mask_module) == 3, "For decoder, each mask should be (mask_att, mask_cross, mask_ffn)"
        else:
            mask_modules = [(False, False, False)] * num_layers
        
        self.layers = bmt.TransformerBlockList([
                bmt.CheckpointBlock(
                    TransformerBlock(
                        dim_model = dim_model, 
                        dim_ff = dim_ff,
                        num_heads = num_heads,
                        dim_head = dim_head,
                        is_decoder = True,
                        dtype = dtype, 
                        int8 = int8,
                        norm_init_var = norm_init_var,
                        norm_bias = norm_bias,
                        norm_eps = norm_eps, 
                        att_init_mean = att_init_mean, 
                        att_init_std = att_init_std,
                        att_bias = att_bias,
                        att_mask_value = att_mask_value,
                        ffn_init_mean = ffn_init_mean, 
                        ffn_init_std = ffn_init_std,
                        ffn_bias = ffn_bias,
                        ffn_activate_fn = ffn_activate_fn,
                        pos_bias_type = pos_bias_type,
                        length_scale = length_scale,
                        attn_scale = attn_scale,
                        dropout_p = dropout_p,
                        parallel_ffn = parallel_ffn,
                        mask_att = mask_modules[ith][0],
                        mask_cross = mask_modules[ith][1],
                        mask_ffn = mask_modules[ith][2],
                    )
                )
                for ith in range(num_layers)
        ])

        self.output_layernorm = LayerNorm(
                    dim_norm = dim_model, 
                    bias = norm_bias, 
                    dtype = dtype,
                    eps = norm_eps, 
                    init_var = norm_init_var)

    def forward(self, hidden_states : torch.Tensor,
                      attention_mask : torch.Tensor,
                      position_bias : torch.Tensor,
                      cross_hidden_states = None,
                      cross_attention_mask = None,
                      cross_position_bias = None,
                      use_cache : bool = False,
                      past_key_values = None,
        ):
        """
        Args:
            hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_dec, dim_model)``): Input of decoder, Can be the embedding of a batch of sequences. 
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_dec, seq_dec)``): Avoid invalid areas to participate in the calculation. 
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, seq_dec, seq_dec)``) Provides position information to attention mechanism. 
            cross_hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_enc, dim_model)``): Input of decoder, Can be the output of encoder. 
            cross_attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_dec, seq_enc)``): Avoid invalid areas to participate in the calculation when the output of encoder participates in the calculation. 
            cross_position_bias(:obj:`torch.Tensor` of shape ``(num_heads, seq_dec, seq_enc)``) Provides position information to attention mechanism when the output of encoder participates in the calculation.  

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_dec, dim_model)``: The decoder output. 

        """
        if not use_cache:
            hidden_states, mid = self.layers(hidden_states, attention_mask, position_bias,
                                        cross_hidden_states, cross_attention_mask, cross_position_bias, return_hidden_states=True)
            hidden_states = self.output_layernorm(hidden_states)
            return hidden_states, mid
        else:
            with torch.no_grad():
                current_key_values = []
                for i, module in enumerate(self.layers):
                    hidden_states  = module(hidden_states, attention_mask, position_bias, 
                                            cross_hidden_states, cross_attention_mask, cross_position_bias,
                                            past_key_value = past_key_values[i] if past_key_values else None, 
                                            use_cache = use_cache)
                    current_key_values.append(hidden_states[1])
                    hidden_states = hidden_states[0]
                hidden_states = self.output_layernorm(hidden_states)
                return hidden_states, current_key_values
