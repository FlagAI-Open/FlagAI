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
from .config import Config

class CPM3Config(Config):

    def __init__(self, vocab_size = 30720,
                        dim_model = 4096,
                        num_heads = 64,
                        dim_head = 64,
                        dim_ff = 10240,
                        num_layers = 32,
                        dropout_p = 0.0,
                        emb_init_mean = 0.0,
                        emb_init_std = 1.0,
                        pos_bias_type = "relative",
                        position_bias_num_buckets = 512,
                        position_bias_max_distance = 2048,
                        pos_init_mean = 0.0,
                        pos_init_std = 1.0,
                        norm_init_var = 1.0,
                        norm_bias = False,
                        norm_eps = 1e-6,
                        att_init_mean = 0.0, 
                        att_init_std = 1.0,
                        att_bias = False,
                        att_mask_value = float("-inf"),
                        ffn_init_mean = 0.0, 
                        ffn_init_std = 1.0,
                        ffn_bias = False,
                        ffn_activate_fn = "gated_gelu",
                        proj_init_mean = 0.0,
                        proj_init_std = 1.0,
                        proj_bias = False,
                        length_scale = True,
                        attn_scale = True,
                        half = True, 
                        int8 = False,
                        tied = True,
                        prompt_types = 32,
                        prompt_length = 64, 
                        segment_types = 34,
                        max_exact_rate = 0.25,
                        max_distance_rate = 1.0,
                        absolute_inner_segment = True,
                        cls_head = None,
                        post_layer_norm=False,
        ):

        super().__init__()
        self.prompt_types = prompt_types
        self.prompt_length = prompt_length
        self.segment_types = segment_types
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.position_bias_num_buckets = position_bias_num_buckets
        self.position_bias_max_distance = position_bias_max_distance
        self.dropout_p = dropout_p
        self.norm_eps = norm_eps
        self.norm_init_var = norm_init_var
        self.emb_init_mean = emb_init_mean
        self.emb_init_std = emb_init_std
        self.att_init_mean = att_init_mean
        self.att_init_std = att_init_std
        self.ffn_init_mean = ffn_init_mean
        self.ffn_init_std = ffn_init_std
        self.length_scale = length_scale
        self.absolute_inner_segment = absolute_inner_segment
        self.max_distance_rate = max_distance_rate
        self.max_exact_rate = max_exact_rate
        self.int8 = int8
        self.tied = tied
        if half: 
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.cls_head = cls_head
        self.vocab_size = vocab_size
        self.pos_bias_type = pos_bias_type
        self.pos_init_mean = pos_init_mean
        self.pos_init_std = pos_init_std
        self.norm_bias = norm_bias
        self.att_bias = att_bias
        self.att_mask_value = att_mask_value
        self.ffn_bias = ffn_bias
        self.ffn_activate_fn = ffn_activate_fn
        self.proj_init_mean = proj_init_mean
        self.proj_init_std = proj_init_std
        self.proj_bias = proj_bias
        self.attn_scale = attn_scale
        self.post_layer_norm = post_layer_norm