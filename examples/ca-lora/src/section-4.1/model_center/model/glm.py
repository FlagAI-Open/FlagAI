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
from ..layer import Encoder, Decoder, Linear, Embedding, RelativePositionEmbedding, LayerNorm
from .basemodel import BaseModel
from .config import GLMConfig

class GLM(BaseModel):

    _CONFIG_TYPE = GLMConfig
    
    def __init__(self, config: GLMConfig):
        
        super().__init__()
        self.sop_tok_id = config.sop_tok_id
        self.mask_tok_id = config.mask_tok_id
        self.encoder = Encoder(
            num_layers = config.num_layers,
            dim_model = config.dim_model, 
            dim_ff = config.dim_ff,
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
            att_mask_value = float(config.att_mask_value),
            pos_bias_type = config.pos_bias_type,
            ffn_init_mean = config.ffn_init_mean, 
            ffn_init_std = config.ffn_init_std,
            ffn_bias = config.ffn_bias,
            ffn_activate_fn = config.ffn_activate_fn,
            length_scale = config.length_scale,
            attn_scale = config.attn_scale,
            dropout_p = config.dropout_p,
        )

        self.input_embedding = Embedding(
            vocab_size = config.vocab_size, 
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )

        self.position_embedding = Embedding(
            vocab_size = config.position_size, 
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )

        self.block_position_embedding = Embedding(
            vocab_size = config.position_size, 
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )

        self.embed_dropout = torch.nn.Dropout(config.dropout_p)
        
        self.tied = config.tied
        self.cls_head = config.cls_head
        if self.cls_head:
            self.output_projection = Linear(
                dim_out = self.cls_head,
                dim_in = config.dim_model,
                length_scale = config.length_scale,
                dtype = config.dtype,
                int8 = config.int8,
                init_mean = config.proj_init_mean,
                init_std = config.proj_init_std,
                bias = config.proj_bias,
            )

    def forward(self, input_ids : torch.Tensor, # (batch, seqlen)
                      position_ids : torch.Tensor = None, # (batch, seqlen)
                      block_position_ids : torch.Tensor = None, # (batch, seqlen)
                      sep : torch.Tensor = None, # (batch)
    ):

        batch = input_ids.size(0)
        seq_len = input_ids.size(1)
        device = input_ids.device
        if sep is None:
            sep = torch.empty(batch, dtype=torch.long, device=device)
            for b in range(batch):
                sep_idx = (input_ids[b]==self.sop_tok_id).nonzero()
                if min(sep_idx.shape) > 0:
                    sep[b] = sep_idx[0]
                else:
                    sep[b] = seq_len
        with torch.no_grad():
            mask_1d = torch.arange(seq_len, device=device)[None, :].repeat(batch, 1) < sep[:, None]
            directional_mask_2d = torch.arange(seq_len, device=device) <= torch.arange(seq_len, device=device).view(-1, 1)
            attention_mask = mask_1d.view(batch, 1, seq_len) | directional_mask_2d.view(1, seq_len, seq_len)

        hidden_states = self.input_embedding(input_ids)
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device)[None, :].repeat(batch, 1)
            for b in range(batch):
                head = sep[b]
                mask_idx = (input_ids[b]==self.mask_tok_id).nonzero() 
                if min((input_ids[b] == self.sop_tok_id).nonzero().shape) > 0:
                    for idx,tail in enumerate((input_ids[b] == self.sop_tok_id).nonzero()[1:]):
                        position_ids[b][head:tail] = mask_idx[idx] 
                        head = tail
                if head != seq_len and min(mask_idx.shape)>0:
                    position_ids[b][head:] = mask_idx[-1]
        if block_position_ids is None:
            block_position_ids = torch.zeros((batch, seq_len), device=device, dtype=torch.long)
            for b in range(batch):
                head = sep[b]
                if min((input_ids[b] == self.sop_tok_id).nonzero().shape) > 1:
                    for tail in (input_ids[b] == self.sop_tok_id).nonzero()[1:]:
                        block_position_ids[b][head:tail] = torch.arange(1, 1+tail-head, device=device) 
                        head = tail
                if head != seq_len:
                    block_position_ids[b][head:] = torch.arange(1, 1+seq_len-head, device=device)
        position_embedding = self.position_embedding(position_ids)
            
        position_embeds = self.position_embedding(position_ids)
        hidden_states = hidden_states + position_embeds
        
        block_position_embeds = self.block_position_embedding(block_position_ids)
        hidden_states = hidden_states + block_position_embeds

        hidden_states = self.embed_dropout(hidden_states)

        hidden_states = self.encoder(hidden_states, attention_mask)

        if self.cls_head:
            logits = self.output_projection(hidden_states)
        else:
            logits = self.input_embedding.projection(hidden_states)

        return logits