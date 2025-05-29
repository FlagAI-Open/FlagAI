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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import bmtrain as bmt

class RelativePositionEmbedding(bmt.DistributedModule):
    """ Relative Position Embedding <https://arxiv.org/abs/1803.02155>

    Args:
        num_heads (int): number of heads used in attention module.
        num_buckets (int, optional): Defaults to 32.
        max_distance (int, optional): Defaults to 128.
        bidirectional (bool, optional): Defaults to False.
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): Defaults to 0.0.
        init_std (float, optional): Defaults to 1.
    """

    def __init__(self, num_heads : int, 
                       num_buckets : int = 32, 
                       max_distance : int = 128, 
                       bidirectional : bool = False, 
                       dtype = torch.half,
                       init_mean : float = 0.0,
                       init_std : float = 1,
                    ):

        super().__init__()

        self.relative_attention_bias = bmt.DistributedParameter(
            torch.empty(num_buckets, num_heads, dtype = dtype), 
            init_method = bmt.ParameterInitializer(nn.init.normal_, mean = init_mean, std = init_std)
        )
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

    def forward(self, query, key):
        """ Provides relative position embeddings for key and query of `num_heads` attention heads. 

        Args:
            query (:obj:`int`): Length of query or query tensor.  
            key (:obj:`int`): Length of key or key tenser.
        Return:
            :obj:`torch.Tensor` of shape ``(num_heads, query_len, key_len)``: Relative position embedding.
        """

        part_buckets = self.num_buckets // (2 if self.bidirectional else 1)
        exact_buckets = part_buckets // 2
        log_buckets = part_buckets - exact_buckets

        if isinstance(query, int):
            query = torch.arange(query, dtype=torch.long, device="cuda")
        if isinstance(key, int):
            key = torch.arange(key, dtype=torch.long, device="cuda")

        if query.dim() == 1:
            relative_position = query[:, None] - key[None, :]
        else:
            relative_position = query[:, :, None] - key[:, None, :]

        neg_pos = relative_position < 0
        relative_position = relative_position.abs()

        small_pos = relative_position < exact_buckets

        log_pos = (torch.clamp(
            torch.log(relative_position.float() / exact_buckets) / math.log(self.max_distance / exact_buckets),
            0,
            0.9999
        ) * log_buckets).long() + exact_buckets

        buckets = torch.where(small_pos, relative_position, log_pos)
        if self.bidirectional:
            buckets = torch.where(
                neg_pos,
                buckets + part_buckets,
                buckets
            )
        else:
            buckets = torch.masked_fill(
                buckets,
                neg_pos,
                0,
            )
        if query.dim() == 1:
            return F.embedding(buckets, self.relative_attention_bias, padding_idx = -1).permute(2, 0, 1).contiguous()
        else:
            return F.embedding(buckets, self.relative_attention_bias, padding_idx = -1).permute(-1, -3, -2).contiguous()


class RotaryEmbedding(nn.Module):
    """`Rotary Position Embedding <https://arxiv.org/abs/2104.09864v2>

    Args:
        rotary_dim (int): rotary dimension
    """
    def __init__(self, rotary_dim: int):
        # Implementation reference https://github.com/huggingface/transformers/blob/master/src/transformers/models/gptj/modeling_gptj.py
        super().__init__()
        self.rotary_dim = rotary_dim

    def fixed_pos_embedding(self, x, seq_len=None, dtype = torch.float):
        dim = x.shape[-1]
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
        sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq).to(x.device).to(dtype)
        return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

    def rotate_every_two(self, x):
        if x.dim() == 4:
            x1 = x[:, :, :, ::2]
            x2 = x[:, :, :, 1::2]
        else:
            x1 = x[:, :, ::2]
            x2 = x[:, :, 1::2]
        x = torch.stack((-x2, x1), axis=-1)
        return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

    def apply_rotary_pos_emb(self, x, sincos, offset=0):
        sin, cos = map(lambda t: t[None, offset : x.shape[-2] + offset, :].repeat_interleave(2, 2), sincos)
        # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
        return (x * cos) + (self.rotate_every_two(x) * sin)

    def forward(self, h_q, h_k):
        """
        Args:
            h_q : (batch_size, num_head, len_q, dim_head)
            h_k : (batch_size, k_num_head, len_k, dim_head)

        Return:
            h_q : (batch_size, num_head, len_q, dim_head)
            h_k : (batch_size, k_num_head, len_k, dim_head)
        """
        if h_q.dim() == 4:
            q_rot = h_q[:, :, :, :self.rotary_dim]
            q_pass = h_q[:, :, :, self.rotary_dim:]
            k_rot = h_k[:, :, :, :self.rotary_dim]
            k_pass = h_k[:, :, :, self.rotary_dim:]
        else:
            q_rot = h_q[:, :, :self.rotary_dim]
            q_pass = h_q[:, :, self.rotary_dim:]
            k_rot = h_k[:, :, :self.rotary_dim]
            k_pass = h_k[:, :, self.rotary_dim:]            

        seq_len = h_k.shape[-2]
        sincos = self.fixed_pos_embedding(k_rot, seq_len=seq_len, dtype=h_k.dtype)
        k_rot = self.apply_rotary_pos_emb(k_rot, sincos, offset=0)
        q_rot = self.apply_rotary_pos_emb(q_rot, sincos, offset=0)

        h_q = torch.cat([q_rot, q_pass], dim=-1)
        h_k = torch.cat([k_rot, k_pass], dim=-1)
        return h_q, h_k


class SegmentPositionEmbedding(bmt.DistributedModule):

    def __init__(self, num_heads,
                       num_segments = 1,
                       num_buckets = 32, 
                       max_distance = 128, 
                       bidirectional = False, 
                       dtype = torch.half,
                       absolute_inner_segment = True,
                       init_mean : float = 0.0,
                       init_std : float = 1
                    ):

        super().__init__()
        
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.num_segments = num_segments
        self.absolute_inner_segment = absolute_inner_segment

        self.relative_attention_bias = bmt.DistributedParameter(
            torch.empty(num_segments * num_segments + num_buckets, num_heads, dtype = dtype), 
            init_method = bmt.ParameterInitializer(nn.init.normal_, mean = init_mean, std = init_std)
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

            assert key_pos is not None or key_segment is not None
            assert query_pos is not None or query_segment is not None

            if isinstance(key_pos, int):
                key_pos = torch.arange(key_pos, dtype=torch.long, device="cuda")[None, :]
            if isinstance(query_pos, int):
                query_pos = torch.arange(query_pos, dtype=torch.long, device="cuda")[None, :]

            if key_pos is not None:
                batch = key_pos.size(0)
                keylen = key_pos.size(1)
                querylen = query_pos.size(1)
            else:
                batch = key_segment.size(0)
                keylen = key_segment.size(1)
                querylen = query_segment.size(1)

            if key_segment is None:
                key_segment = torch.zeros(keylen, dtype=torch.long, device="cuda")[None, :]
            if query_segment is None:
                query_segment = torch.zeros(querylen, dtype=torch.long, device="cuda")[None, :]
            if key_pos is None:
                key_pos = torch.arange(keylen, dtype=torch.long, device="cuda")[None, :]
            if query_pos is None:
                query_pos = torch.arange(querylen, dtype=torch.long, device="cuda")[None, :]

            assert key_pos.size(0) == query_pos.size(0)
            assert keylen == key_segment.size(1) and querylen == query_segment.size(1)

            key_pos = key_pos.view(batch, -1, keylen)
            query_pos = query_pos.view(batch, querylen, -1)
            key_segment = key_segment.view(batch, -1, keylen)
            query_segment = query_segment.view(batch, querylen, -1)

            relative_position_bucket = self._segment_relative_position_bucket(query_segment, key_segment)
            relative_position_bucket = relative_position_bucket + self.num_buckets  # 与相对位置编码区间不重叠
            # b*q*k
            if self.absolute_inner_segment:
                absolute_position_bucket = self._absolute_position_bucket(
                    key_pos - query_pos,
                    bidirectional=self.bidirectional,
                    num_buckets=self.num_buckets,
                    max_distance = self.max_distance
                )
                relative_position_bucket = relative_position_bucket.to(torch.int32)
                relative_position_bucket = torch.where((key_segment == query_segment), absolute_position_bucket, relative_position_bucket)
            # (batch, len_q, len_k)

        # (batch, len_q, len_k, num_heads)
        embeds = F.embedding(relative_position_bucket, self.relative_attention_bias)
        
        # (batch, num_heads, len_q, len_k)
        embeds = embeds.permute(0, 3, 1, 2).contiguous()
        return embeds

    def _segment_relative_position_bucket(self, query_segment, key_segment):
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
