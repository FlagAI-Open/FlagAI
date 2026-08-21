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
from .basemodel import BaseModel
from .config import VitConfig
from ..layer import PatchEmbedding, Encoder, Linear

class ViT(BaseModel):

    _CONFIG_TYPE = VitConfig
    def __init__(self, config: VitConfig):

        super().__init__()

        hidden_size = config.hidden_size
        self.num_features =  config.hidden_size  # num_features for consistency with other models
        self.patch_embed = PatchEmbedding(
                img_size=config.img_size, 
                patch_size=config.patch_size, 
                in_chans=config.channels_in,
                embed_dim=hidden_size, dtype=config.dtype)
        self.num_patches = self.patch_embed.num_patches

        self.pos_drop = torch.nn.Dropout(p=config.drop)
        self.representation_size = config.representation_size

        self.blocks = Encoder(num_layers=config.num_layers,
                               dim_model=hidden_size,dim_ff=config.mlp_size,
                               num_heads=config.num_heads,
                               dim_head=hidden_size//config.num_heads,
                               att_bias=config.attn_bias,
                               attn_scale=True, 
                               dropout_p=config.drop,
                               norm_bias=config.norm_bias,
                               ffn_bias=config.ffn_bias,
                               ffn_activate_fn="gelu",
                               dtype=config.dtype)

        if self.representation_size is not None:
            self.representation_layer = Linear(hidden_size,config.representation_size)
            hidden_size = config.representation_size

        self.head = Linear(hidden_size, config.num_classes, dtype=config.dtype,bias=True)

    def forward(self, input_seq, register_blk=-1, attention_mask=None):
        batch = input_seq.shape[0]
        hidden_state = self.patch_embed(input_seq)
        device = input_seq.device
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)
        else:
            attention_mask = torch.ones(self.num_patches+1, device=device,dtype=torch.int32)[None, :].repeat(batch, 1)
        attention_mask = attention_mask.view(batch, self.num_patches+1, 1) & attention_mask.view(batch, 1, self.num_patches+1)
        hidden_state = self.pos_drop(hidden_state)
        hidden_state = self.blocks(hidden_state,attention_mask=attention_mask)
        if self.representation_size is not None:
            hidden_state = self.representation_layer(hidden_state)
            hidden_state = torch.tanh(hidden_state)
        logits = self.head(hidden_state[:,0])
        return logits
