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

class VitConfig(Config):
    """
    This is a configuration class that stores the configuration of the Vit model, which inherits from the Config class.
    It is used to instantiate the vit model according to the specified parameters and define the model architecture.
    You can set specific parameters to control the output of the model.

    For example:
    [`hidden_size`] is used to determine the Dimension of the encoder layers.
    You can choose to use the default value of 768 or customize their dimensions.  
    
    """
    def __init__(self, img_size=224,
                       patch_size=16,
                       channels_in=3,
                       num_classes=1000,
                       hidden_size=768,
                       num_layers=12,
                       num_heads=12,
                       mlp_size=3072,
                       attn_bias=True,
                       attn_scale=None,
                       norm_bias=True,
                       ffn_bias=True,
                       representation_size=None,
                       drop=0.,
                       half=True,
                       dtype=torch.float):

        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.channels_in = channels_in
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_size = mlp_size
        self.attn_bias = attn_bias
        self.attn_scale = attn_scale
        self.norm_bias = norm_bias
        self.ffn_bias = ffn_bias
        self.representation_size = representation_size
        self.drop = drop
        if half:
            self.dtype = torch.half
        else:
            self.dtype = torch.float