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
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn as nn
import os
from .layer_norm import BertLayerNorm

import math

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
except:
    from .layer_norm import LayerNorm
from flagai.mpu.initialize import get_model_parallel_rank
from flagai.mpu.initialize import get_model_parallel_world_size
from flagai.mpu.mappings import copy_to_model_parallel_region
from flagai.mpu.mappings import gather_from_model_parallel_region
from flagai.mpu.mappings import reduce_from_model_parallel_region
from flagai.mpu.mappings import scatter_to_model_parallel_region
from flagai.mpu.utils import divide
from flagai.mpu.utils import VocabUtility
from flagai.model.utils import normal_init_method


class PositionalEmbedding(torch.nn.Module):

    def __init__(self, hidden_size):
        super(PositionalEmbedding, self).__init__()

        self.hidden_size = hidden_size

        inv_freq = 1 / (10000
                        **(torch.arange(0.0, hidden_size, 2.0) / hidden_size))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


class WordEmbedding(nn.Module):
    """
    input embeddin only has word embedding
    """

    def __init__(self, args, vocab_size):
        super(WordEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, _):
        emb = self.word_embedding(src)
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


def _initialize_affine_weight(weight,
                              output_size,
                              input_size,
                              per_partition_size,
                              partition_dim,
                              init_method,
                              stride=1,
                              return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""
    # If we only use 1 process for model parallelism, bypass scatter.
    if os.getenv("ENV_TYPE") == 'deepspeed+mpu':
        world_size = get_model_parallel_world_size()
    else:
        world_size = 1
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None

    # Initialize master weight
    master_weight = torch.empty(output_size,
                                input_size,
                                dtype=weight.dtype,
                                requires_grad=False)
    init_method(master_weight)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight,
                              per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_model_parallel_rank()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        # Divide the weight matrix along the vocaburaly dimension.
        if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
            self.vocab_start_index, self.vocab_end_index = \
                VocabUtility.vocab_range_from_global_vocab_size(
                    self.num_embeddings, get_model_parallel_rank(),
                    get_model_parallel_world_size())
        else:
            self.vocab_start_index = 0
            self.vocab_end_index = self.num_embeddings

        self.num_embeddings_per_partition = self.vocab_end_index - \
                                            self.vocab_start_index

        # Allocate weights.
        self.weight = Parameter(
            torch.Tensor(self.num_embeddings_per_partition,
                         self.embedding_dim))
        if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
            self.weight.model_parallel = True
        # And initialize.

        _initialize_affine_weight(self.weight, self.num_embeddings,
                                  self.embedding_dim,
                                  self.num_embeddings_per_partition, 0,
                                  init_method)

    def forward(self, input_):
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)

        # Mask the output embedding.
        output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
            output = reduce_from_model_parallel_region(output_parallel)
            return output
        return output_parallel


class ParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the embedding dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 init_method=init.xavier_normal_,
                 keep_master_weight_for_test=False):
        super(ParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set some detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        # Divide the weight matrix along the embedding dimension.
        if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
            world_size = get_model_parallel_world_size()
        else:
            world_size = 1
        self.embedding_dim_per_partition = divide(self.embedding_dim,
                                                  world_size)

        # Allocate weights.
        self.weight = Parameter(
            torch.Tensor(self.num_embeddings,
                         self.embedding_dim_per_partition))
        if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
            self.weight.model_parallel = True
        # And initialize.
        _initialize_affine_weight(self.weight,
                                  self.num_embeddings,
                                  self.embedding_dim,
                                  self.embedding_dim_per_partition,
                                  1,
                                  init_method,
                                  stride=1,
                                  return_master_weight=False)

    def forward(self, input_):
        if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
            input_parallel = copy_to_model_parallel_region(input_)
        else:
            input_parallel = input_
        output_parallel = F.embedding(input_parallel, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)

        if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
            output = gather_from_model_parallel_region(output_parallel)
            return output
        return output_parallel


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, vocab_size, hidden_size, initializer_range,
                 max_position_embeddings, type_vocab_size, layernorm_epsilon,
                 hidden_dropout_prob):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = VocabParallelEmbedding(
            vocab_size,
            hidden_size,
            init_method=normal_init_method(mean=0.0, std=initializer_range))
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(hidden_size, eps=layernorm_epsilon)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length,
                                    dtype=torch.long,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)

        embeddings = self.dropout(embeddings)
        return embeddings


class CPM3Embedding(torch.nn.Module):
    r"""Embed a sequence of indices through a embedding lookup matrix :math:`\mathbf{W}`.

    Args:
        vocab_size (int): indices be in range :math:`[0, \text{vocab_size})`
        embedding_size (int): the output dimension of the embedding lookup matrix.
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)`. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)`. Defaults to 1.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        length_scale: bool = False,
        dtype=torch.half,
        int8: bool = False,
    ):
        super().__init__()
        self.dim_model = embedding_size
        self.weight = torch.nn.Parameter(
            torch.empty(vocab_size, embedding_size, dtype=dtype), )
        self.length_scale = length_scale
        self.int8 = int8

    def forward(self, ids: torch.Tensor):
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

    def projection(self, x: torch.Tensor):
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


class CPM3SegmentPositionEmbedding(torch.nn.Module):

    def __init__(self,
                 num_heads,
                 num_segments=1,
                 num_buckets=32,
                 max_distance=128,
                 max_exact_rate=0.25,
                 max_distance_rate=1.0,
                 bidirectional=False,
                 dtype=torch.half,
                 absolute_inner_segment=True):

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
        self.relative_attention_bias = torch.nn.Parameter(
            torch.empty(num_segments * num_segments + num_buckets,
                        num_heads,
                        dtype=dtype))

    def forward(self,
                key_pos=None,
                query_pos=None,
                key_segment=None,
                query_segment=None):
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
            assert keylen == key_segment.size(
                1) and querylen == query_segment.size(1)

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
            relative_position_bucket = self._segment_relative_position_bucket(
                query_segment, key_segment)
            relative_position_bucket = relative_position_bucket + self.num_buckets  # 与相对位置编码区间不重叠

            # bucket_segment = (key_segment != query_segment) * (key_segment + query_segment * (self.num_segments - 1) + (query_segment >= key_segment))
            # relative_position_bucket += bucket_segment * self.num_buckets
            # b*q*k
            if self.absolute_inner_segment:
                absolute_position_bucket = self._absolute_position_bucket(
                    torch.arange(
                        keylen,
                        dtype=torch.int32,
                        device=relative_position_bucket.device)[None, :] -
                    torch.arange(querylen,
                                 dtype=torch.int32,
                                 device=relative_position_bucket.device)[:,
                                                                         None],
                    bidirectional=self.bidirectional,
                    num_buckets=self.num_buckets,
                    max_distance=self.max_distance)
                relative_position_bucket = torch.where(
                    (key_segment == query_segment),
                    absolute_position_bucket[None, :, :],
                    relative_position_bucket)
            # (batch, len_q, len_k)

        # (batch, len_q, len_k, num_heads)
        embeds = F.embedding(relative_position_bucket,
                             self.relative_attention_bias)
        # (batch, num_heads, len_q, len_k)
        embeds = embeds.permute(0, 3, 1, 2).contiguous()
        return embeds

    def _relative_position_bucket(self,
                                  relative_position,
                                  bidirectional=True,
                                  num_buckets=32,
                                  max_exact=0.125,
                                  max_distance=0.5):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            max_exact /= 2
            relative_buckets = (relative_position > 0).to(
                torch.int32) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position,
                                           torch.zeros_like(relative_position))
        max_exact /= 2
        is_small = relative_position < max_exact
        half_num_buckets = num_buckets // 2
        relative_postion_if_large = half_num_buckets + (
            torch.log(relative_position / max_exact) /
            math.log(max_distance / max_exact) *
            (num_buckets - half_num_buckets)).to(torch.int32)
        relative_postion_if_large = torch.min(
            relative_postion_if_large,
            torch.full_like(relative_postion_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small,
                                        (relative_position / max_exact *
                                         half_num_buckets).to(torch.int32),
                                        relative_postion_if_large)
        return relative_buckets

    def _segment_relative_position_bucket(self, query_segment, key_segment):
        """
            All positional encodings of segment2 are the same in the view of segment1. Same for segment3 and segment1(but is a different value)
        """
        return query_segment * self.num_segments + key_segment

    def _absolute_position_bucket(self,
                                  relative_position,
                                  bidirectional=True,
                                  num_buckets=32,
                                  max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets = (relative_position > 0).to(
                torch.int32) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position,
                                           torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) /
            math.log(max_distance / max_exact) *
            (num_buckets - max_exact)).to(torch.int32)
        relative_postion_if_large = torch.min(
            relative_postion_if_large,
            torch.full_like(relative_postion_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small,
                                        relative_position.to(torch.int32),
                                        relative_postion_if_large)
        return relative_buckets
