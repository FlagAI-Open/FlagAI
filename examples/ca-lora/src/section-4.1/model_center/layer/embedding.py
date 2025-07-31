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
import math
import torch.nn.functional as F
from .conv import Conv2d
from .conv import to_2tuple 
try:
    from torch import _assert
except ImportError:
    def _assert(condition:bool, message:str):
        assert condition, message
        
class Embedding(bmt.DistributedModule):
    r"""Embed a sequence of indices through a embedding lookup matrix :math:`\mathbf{W}`.

    Args:
        vocab_size (int): indices be in range :math:`[0, \text{vocab_size})`
        embedding_size (int): the output dimension of the embedding lookup matrix.
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)`. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)`. Defaults to 1.
    """
    def __init__(self,
                 vocab_size : int,
                 embedding_size : int,
                 length_scale : bool = False,
                 dtype = torch.half,
                 int8 :bool = False,
                 init_mean : float = 0.0,
                 init_std : float= 1,
                 padding_idx : int = None,
                ):
        super().__init__()
        self.dim_model = embedding_size
        self.weight = bmt.DistributedParameter(
            torch.empty(vocab_size, embedding_size, dtype=dtype),
            init_method = bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        )
        self.padding_idx = padding_idx
        if self.padding_idx is not None:
            if self.padding_idx > 0:
                assert self.padding_idx < vocab_size, "padding_idx must be less than vocab_size"
            elif self.padding_idx < 0:
                assert self.padding_idx >= -vocab_size, "padding_idx must be greater than or equal to -vocab_size"
                self.padding_idx = vocab_size + self.padding_idx
            with torch.no_grad():
                self.weight.data[self.padding_idx].fill_(0)

        self.length_scale = length_scale
        self.int8 = int8

    def forward(self, ids : torch.Tensor):
        """ 
        Args:
            ids (:obj:`torch.Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.

        Return:
            :obj:`torch.Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """
        
        embeds = F.embedding(ids, self.weight,self.padding_idx)
        if self.length_scale:
            embeds = embeds / math.sqrt(self.dim_model)
        return embeds
    
    def projection(self, x : torch.Tensor):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.

        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, vocab_output_size)``: The projection output.
        """
        if self.length_scale:
            x = x / math.sqrt(self.dim_model)
        logits = F.linear(x, self.weight)
        return logits

class PatchEmbedding(bmt.DistributedModule):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,
                img_size=224,
                patch_size=16,
                in_chans=3,
                embed_dim=768,
                flatten=True,
                dtype=torch.half
                ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, dtype=dtype)
        self.cls_token = bmt.DistributedParameter(torch.empty((1,1,embed_dim), dtype=dtype))
        self.pos_embed = bmt.DistributedParameter(torch.empty((1,self.num_patches+1,embed_dim), dtype=dtype))
    def forward(self, x):
        B,C,H,W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, 1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:,:x.size(1),:]
        return x