import torch
from flagai.model.layers.attentions import CPM3SelfAttention
from flagai.model.layers.attentions_bmt import CPM3bmtSelfAttention
from flagai.model.layers.attentions import CPM3CrossAttention
from flagai.model.layers.attentions_bmt import CPM3bmtCrossAttention
from flagai.model.layers.feedforward import CPM3FFN
from flagai.model.layers.feedforward_bmt import CPM3bmtFFN
from typing import *


class CPM3Block(torch.nn.Module):
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

    def __init__(self, config):
        super().__init__()

        self.is_decoder = config.is_decoder

        self.self_att = CPM3SelfAttention(
            dim_model = config.dim_model, 
            num_heads = config.num_heads, 
            dim_head = config.dim_head, 
            dtype = config.dtype,
            int8 = config.int8, 
            norm_eps = config.norm_eps, 
            norm_init_var = config.norm_init_var,
            norm_bias = config.norm_bias,
            att_init_mean = config.att_init_mean, 
            att_init_std = config.att_init_std,
            att_bias = config.att_bias,
            att_mask_value = config.att_mask_value,
            pos_bias_type = config.pos_bias_type,
            post_layer_norm = config.post_layer_norm,
            length_scale = config.length_scale,
            attn_scale = config.attn_scale,
            dropout_p = config.dropout_p,
        )

        if self.is_decoder:
            self.cross_att = CPM3CrossAttention(
                dim_model = config.dim_model, 
                num_heads = config.num_heads, 
                dim_head = config.dim_head, 
                dtype = config.dtype,
                int8 = config.int8, 
                norm_eps = config.norm_eps, 
                norm_init_var = config.norm_init_var,
                norm_bias = config.norm_bias,
                att_init_mean = config.att_init_mean, 
                att_init_std = config.att_init_std,
                att_bias = config.att_bias,
                att_mask_value = config.att_mask_value,
                pos_bias_type = config.pos_bias_type,
                length_scale = config.length_scale,
                attn_scale = config.attn_scale,
                dropout_p = config.dropout_p,
            )
        else:
            self.cross_att = None

        self.ffn = CPM3FFN(
            dim_model = config.dim_model, 
            dim_ff = config.dim_ff,
            dtype = config.dtype, 
            int8 = config.int8,
            norm_eps = config.norm_eps, 
            norm_init_var = config.norm_init_var,
            norm_bias = config.norm_bias,
            ffn_init_mean = config.ffn_init_mean, 
            ffn_init_std = config.ffn_init_std,
            ffn_bias = config.ffn_bias,
            ffn_activate_fn = config.ffn_activate_fn,
            length_scale = config.length_scale,
            dropout_p = config.dropout_p,
            post_layer_norm = config.post_layer_norm,
        )

        self.parallel_ffn = config.parallel_ffn

    def forward(self,
                self_hidden_states : torch.Tensor,
                self_attention_mask : torch.Tensor,
                self_position_bias : Optional[torch.Tensor] = None,
                cross_hidden_states = None,
                cross_attention_mask = None,
                cross_position_bias = None,
                past_key_value = None
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
        # (batch, dim_model, seq_self)
        hidden_states, present_key_value = self.self_att(self_hidden_states,
                                                        attention_mask = self_attention_mask,
                                                        position_bias = self_position_bias,
                                                        past_key_value = past_key_value)

        # (batch, dim_model, seq_self)
        # TODO: use cache in cross_att
        # if self.is_decoder and self.cross_att is not None:
        #     hidden_states = self.cross_att(hidden_states = hidden_states,
        #                                    key_value_states = cross_hidden_states,
        #                                    attention_mask = cross_attention_mask,
        #                                    position_bias = cross_position_bias)

        # (batch, dim_model, seq_self)
        if self.parallel_ffn:
            hidden_states_2 = self.ffn(self_hidden_states)
            hidden_states = hidden_states - self_hidden_states + hidden_states_2
        else:
            hidden_states = self.ffn(hidden_states)

        return hidden_states, present_key_value

class CPM3bmtBlock(torch.nn.Module):
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

    def __init__(self, config):
        super().__init__()

        self.is_decoder = config.is_decoder

        self.self_att = CPM3bmtSelfAttention(
            dim_model = config.dim_model, 
            num_heads = config.num_heads, 
            dim_head = config.dim_head, 
            dtype = config.dtype,
            int8 = config.int8, 
            norm_eps = config.norm_eps, 
            norm_init_var = config.norm_init_var,
            norm_bias = config.norm_bias,
            att_init_mean = config.att_init_mean, 
            att_init_std = config.att_init_std,
            att_bias = config.att_bias,
            att_mask_value = config.att_mask_value,
            pos_bias_type = config.pos_bias_type,
            post_layer_norm = config.post_layer_norm,
            length_scale = config.length_scale,
            attn_scale = config.attn_scale,
            dropout_p = config.dropout_p,
        )

        if self.is_decoder:
            self.cross_att = CPM3bmtCrossAttention(
                dim_model = config.dim_model, 
                num_heads = config.num_heads, 
                dim_head = config.dim_head, 
                dtype = config.dtype,
                int8 = config.int8, 
                norm_eps = config.norm_eps, 
                norm_init_var = config.norm_init_var,
                norm_bias = config.norm_bias,
                att_init_mean = config.att_init_mean, 
                att_init_std = config.att_init_std,
                att_bias = config.att_bias,
                att_mask_value = config.att_mask_value,
                pos_bias_type = config.pos_bias_type,
                length_scale = config.length_scale,
                attn_scale = config.attn_scale,
                dropout_p = config.dropout_p,
            )
        else:
            self.cross_att = None

        self.ffn = CPM3bmtFFN(
            dim_model = config.dim_model, 
            dim_ff = config.dim_ff,
            dtype = config.dtype, 
            int8 = config.int8,
            norm_eps = config.norm_eps, 
            norm_init_var = config.norm_init_var,
            norm_bias = config.norm_bias,
            ffn_init_mean = config.ffn_init_mean, 
            ffn_init_std = config.ffn_init_std,
            ffn_bias = config.ffn_bias,
            ffn_activate_fn = config.ffn_activate_fn,
            length_scale = config.length_scale,
            dropout_p = config.dropout_p,
            post_layer_norm = config.post_layer_norm,
        )

        self.parallel_ffn = config.parallel_ffn

    def forward(self,
                self_hidden_states : torch.Tensor,
                self_attention_mask : torch.Tensor,
                self_position_bias : Optional[torch.Tensor] = None,
                cross_hidden_states = None,
                cross_attention_mask = None,
                cross_position_bias = None,
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
        # (batch, dim_model, seq_self)
        hidden_states = self.self_att(self_hidden_states,
                                      attention_mask = self_attention_mask,
                                      position_bias = self_position_bias)
        # (batch, dim_model, seq_self)
        if self.is_decoder and self.cross_att is not None:
            hidden_states = self.cross_att(hidden_states = hidden_states,
                                           key_value_states = cross_hidden_states,
                                           attention_mask = cross_attention_mask,
                                           position_bias = cross_position_bias)

        # (batch, dim_model, seq_self)
        if self.parallel_ffn:
            hidden_states_2 = self.ffn(self_hidden_states)
            hidden_states = hidden_states - self_hidden_states + hidden_states_2
        else:
            hidden_states = self.ffn(hidden_states)
        return hidden_states