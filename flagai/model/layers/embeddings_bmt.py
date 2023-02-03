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

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

import torch
import torch.nn.functional as F

import bmtrain as bmt
import math


class CPM3bmtEmbedding(bmt.DistributedModule):
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
                ):
        super().__init__()
        self.dim_model = embedding_size
        self.weight = bmt.DistributedParameter(
            torch.empty(vocab_size, embedding_size, dtype=dtype),
            init_method = bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        )
        self.length_scale = length_scale
        self.int8 = int8

    def forward(self, ids : torch.Tensor):
        """ 
        Args:
            ids (:obj:`torch.Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.

        Return:
            :obj:`torch.Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """
        embeds = F.embedding(ids, self.weight)
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


class CPM3bmtSegmentPositionEmbedding(bmt.DistributedModule):

    def __init__(self, num_heads, 
    	               num_segments = 1,
                       num_buckets = 32, 
                       max_distance = 128, 
                       max_exact_rate = 0.25,
                       max_distance_rate = 1.0,
                       bidirectional = False, 
                       dtype = torch.half,
                       absolute_inner_segment = True):

        super().__init__()
        
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.max_exact_rate = max_exact_rate
        self.max_distance_rate = max_distance_rate
        self.bidirectional = bidirectional
        self.num_segments = num_segments
        self.absolute_inner_segment = absolute_inner_segment
        # self.relative_attention_bias = bmt.DistributedParameter(
        #     torch.empty((num_segments * (num_segments - 1) + 1) * num_buckets, num_heads, dtype=dtype),
        #     init_method = bmt.ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=0.0)
        # )
        self.relative_attention_bias = bmt.DistributedParameter(
            torch.empty(num_segments * num_segments + num_buckets, num_heads, dtype=dtype),
            init_method = bmt.ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=0.0)
        )

    def forward(self, key_pos = None, query_pos = None, key_segment = None, query_segment = None):
        """
        Args:
            key_len: int
            query_len : int
        Returns:
            out : (batch_size, num_heads, query_len, key_len)   fp16
        """
        with torch.no_grad():
        
            batch = key_pos.size(0)
            keylen = key_pos.size(1)
            querylen = query_pos.size(1)

            assert key_pos.size(0) == query_pos.size(0)
            assert keylen == key_segment.size(1) and querylen == query_segment.size(1)

            key_pos = key_pos.view(batch, -1, keylen)
            query_pos = query_pos.view(batch, querylen, -1)
            key_segment = key_segment.view(batch, -1, keylen)
            query_segment = query_segment.view(batch, querylen, -1)

            # relative_position_bucket = self._relative_position_bucket(
            #     key_pos - query_pos,
            #     bidirectional=self.bidirectional,
            #     num_buckets=self.num_buckets,
            #     max_exact = self.max_exact_rate,
            #     max_distance = self.max_distance_rate
            # )
            relative_position_bucket = self._segment_relative_position_bucket(query_segment, key_segment)
            relative_position_bucket = relative_position_bucket + self.num_buckets  # 与相对位置编码区间不重叠

            # bucket_segment = (key_segment != query_segment) * (key_segment + query_segment * (self.num_segments - 1) + (query_segment >= key_segment))
            # relative_position_bucket += bucket_segment * self.num_buckets
            # b*q*k
            if self.absolute_inner_segment:
                absolute_position_bucket = self._absolute_position_bucket(
                    torch.arange(keylen, dtype=torch.int32, device=relative_position_bucket.device)[None, :] - torch.arange(querylen, dtype=torch.int32, device=relative_position_bucket.device)[:, None],
                    bidirectional=self.bidirectional,
                    num_buckets=self.num_buckets,
                    max_distance = self.max_distance
                )
                relative_position_bucket = torch.where((key_segment == query_segment), absolute_position_bucket[None, :, :], relative_position_bucket)
            # (batch, len_q, len_k)
 
        # (batch, len_q, len_k, num_heads)
        embeds = F.embedding(relative_position_bucket, self.relative_attention_bias)
        # (batch, num_heads, len_q, len_k)
        embeds = embeds.permute(0, 3, 1, 2).contiguous()
        return embeds

    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets = 32, max_exact=0.125, max_distance=0.5):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            max_exact /= 2
            relative_buckets = (relative_position > 0).to(torch.int32) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact /= 2
        is_small = relative_position < max_exact
        half_num_buckets = num_buckets // 2
        relative_postion_if_large = half_num_buckets + (torch.log(relative_position / max_exact) / math.log(max_distance / max_exact) * (num_buckets - half_num_buckets)).to(torch.int32)
        relative_postion_if_large = torch.min(relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, (relative_position / max_exact * half_num_buckets).to(torch.int32), relative_postion_if_large)
        return relative_buckets

    def _segment_relative_position_bucket(self, query_segment, key_segment):
        """
            All positional encodings of segment2 are the same in the view of segment1. Same for segment3 and segment1(but is a different value)
        """
        return query_segment * self.num_segments + key_segment

    def _absolute_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets = (relative_position > 0).to(torch.int32) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).to(torch.int32)
        relative_postion_if_large = torch.min(relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_position.to(torch.int32), relative_postion_if_large)
        return relative_buckets
