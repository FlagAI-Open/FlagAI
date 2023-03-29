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
import os
import torch
from torch.nn import Linear
from flagai.model.layers.activations import ACT2FN
from flagai.model.layers.attentions import BertAttention
from flagai.model.layers.feedforward import ColumnParallelLinear
from flagai.model.layers.feedforward import RowParallelLinear
from flagai.model.layers.layer_norm import BertLayerNorm
from flagai.model.utils import normal_init_method


class BertOutput(torch.nn.Module):

    def __init__(self, intermediate_size, hidden_size, layernorm_epsilon,
                 hidden_dropout_prob, initializer_range):
        super(BertOutput, self).__init__()
        if os.getenv("ENV_TYPE") == 'deepspeed+mpu':
            init_method = normal_init_method(mean=0.0, std=initializer_range)
            self.dense = RowParallelLinear(input_size=intermediate_size,
                                           output_size=hidden_size,
                                           bias=True,
                                           input_is_parallel=True,
                                           stride=1,
                                           init_method=init_method)
        else:
            self.dense = Linear(
                intermediate_size,
                hidden_size,
            )

        self.LayerNorm = BertLayerNorm(hidden_size, eps=layernorm_epsilon)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        ln_input = hidden_states + input_tensor

        hidden_states = self.LayerNorm(ln_input)

        return hidden_states


class BertIntermediate(torch.nn.Module):

    def __init__(self, hidden_size, intermediate_size, initializer_range,
                 hidden_act):
        super(BertIntermediate, self).__init__()
        if os.getenv("ENV_TYPE") == 'deepspeed+mpu':
            self.dense = ColumnParallelLinear(input_size=hidden_size,
                                              output_size=intermediate_size,
                                              bias=True,
                                              gather_output=False,
                                              stride=1,
                                              init_method=normal_init_method(
                                                  mean=0.0,
                                                  std=initializer_range))
        else:
            self.dense = Linear(
                hidden_size,
                intermediate_size,
            )
        self.intermediate_act_fn = ACT2FN[hidden_act] \
            if isinstance(hidden_act, str) else hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertBlock(torch.nn.Module):

    def __init__(self, hidden_size, num_attention_heads,
                 attention_probs_dropout_prob, initializer_range,
                 layernorm_epsilon, hidden_dropout_prob, intermediate_size,
                 hidden_act, enable_flash_atten=False):
        super(BertBlock, self).__init__()
        self.attention = BertAttention(hidden_size, num_attention_heads,
                                       attention_probs_dropout_prob,
                                       initializer_range, layernorm_epsilon,
                                       hidden_dropout_prob, enable_flash_atten)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size,
                                             initializer_range, hidden_act)
        self.output = BertOutput(intermediate_size, hidden_size,
                                 layernorm_epsilon, hidden_dropout_prob,
                                 initializer_range)

    def forward(self, hidden_states, attention_mask, **kwargs):
        attention_output = self.attention(hidden_states, attention_mask, **kwargs)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
