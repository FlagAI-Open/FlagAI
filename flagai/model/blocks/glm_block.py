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
"""Transformer."""

import torch

from flagai.model.layers.attentions import ParallelSelfAttention
from flagai.model.layers.attentions import ParallelCrossAttention
from flagai.model.layers.feedforward import MLPForward

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
except:
    from torch.nn import LayerNorm


class GLMBlock(torch.nn.Module):
    """A single layer transformer for GLM.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layers (attention output and
                                  mlp output) initialization. If None,
                                  use `init_method`.
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 init_method,
                 output_layer_init_method=None,
                 relative_encoding=False,
                 performer=False,
                 attention_scale=1.0):
        super(GLMBlock, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = ParallelSelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method,
            relative_encoding=relative_encoding,
            performer=performer,
            attention_scale=attention_scale)

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(hidden_size,
                                                  eps=layernorm_epsilon)

        # MLP
        self.mlp = MLPForward(
            hidden_size,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method)

    def forward(self,
                hidden_states,
                ltor_mask,
                position_embeddings=None,
                r_w_bias=None,
                r_r_bias=None,
                mem=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        mem = self.input_layernorm(mem) if mem is not None else None
        # Self attention.
        attention_output = self.attention(layernorm_output, ltor_mask,
                                          position_embeddings, r_w_bias,
                                          r_r_bias, mem)
        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection.
        output = layernorm_input + mlp_output

        return output


class GLMDecoderBlock(torch.nn.Module):
    """A single layer transformer for GPT2.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layers (attention output and
                                  mlp output) initialization. If None,
                                  use `init_method`.
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 init_method,
                 output_layer_init_method=None):
        super(GLMDecoderBlock, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.self_attention = ParallelSelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method)

        # Layernorm after the self attention.
        self.post_self_layernorm = LayerNorm(hidden_size,
                                             eps=layernorm_epsilon)

        self.cross_attention = ParallelCrossAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method)

        # Layernorm after the cross attention.
        self.post_attention_layernorm = LayerNorm(hidden_size,
                                                  eps=layernorm_epsilon)

        # MLP
        self.mlp = MLPForward(
            hidden_size,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method)

    def forward(self,
                hidden_states,
                encoder_states,
                ltor_mask,
                cross_mask=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        self_attention_output = self.self_attention(layernorm_output,
                                                    ltor_mask)
        # Residual connection.
        self_layernorm_input = hidden_states + self_attention_output
        # Layer norm post the self attention.
        self_layernorm_output = self.post_self_layernorm(self_layernorm_input)
        # Cross attention
        attention_output = self.cross_attention(self_layernorm_output,
                                                encoder_states, cross_mask)
        # Residual connection
        layernorm_input = self_layernorm_input + attention_output
        # Layer norm post the cross attention
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection.
        output = layernorm_input + mlp_output
        return output
