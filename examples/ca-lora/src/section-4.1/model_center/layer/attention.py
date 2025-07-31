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
from typing import Optional

import torch
import bmtrain as bmt
import torch.nn.functional as F
from .linear import Linear


class Attention(bmt.DistributedModule):

    """ Attention module consisting procedure of Q, K, V combination and its output projection. 
    For more detail, see `Attention is All you Need <https://arxiv.org/abs/1706.03762>`_.

    Args:
        dim_in (int): input dimension.
        dim_head (int): dimension of each heads used in attention.
        num_heads (int): number of heads used in attention.
        dim_out (int, optional): output dimension. Defaults to None, which means dim_in = dim_out.
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in attetion module. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in attention module. Defaults to 0.02.
        bias (bool, optional): whether to use bias term in fully-connected layers used in attention module. Defaults to False.
        mask_value (float, optional): mask value of the masked position. Defaults to `-inf`.
        pos_bias_type (str, optional): `relative` for relative position bias, `rotary` for ratery position embedding. Defaults to `none`.
        attn_scale (bool, optional): whether to scale before softmax, i.e., :math:`\text{softmax}({Q K^T \over \sqrt{\text{dim_model}}})`. Default to False.
        dropout_p (float, optional): Defaults to 0.
    """

    def __init__(self, dim_in : int, 
                       dim_head : int,
                       num_heads : int, 
                       dim_out : int = None,
                       dtype = torch.half,
                       int8 = False, 
                       init_mean = 0.0, 
                       init_std = 0.02,
                       bias = False,
                       mask_value : float = float("-inf"),
                       pos_bias_type : str = "none",
                       length_scale : bool = False,
                       attn_scale : bool = False,
                       dropout_p : float= 0,
                       shared_key_and_value = False,
        ):

        super().__init__()

        if dim_out is None:
            dim_out = dim_in

        num_heads_kv = 1 if shared_key_and_value else num_heads 

        self.project_q = Linear(
            dim_in = dim_in,
            dim_out = num_heads * dim_head,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        self.project_k = Linear(
            dim_in = dim_in,
            dim_out = num_heads_kv * dim_head,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        self.project_v = Linear(
            dim_in = dim_in,
            dim_out = num_heads_kv * dim_head,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        self.attention_out = Linear(
            dim_in = num_heads * dim_head,
            dim_out = dim_out,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )
        self.init_mean = init_mean
        self.init_std = init_std
        self.dim_in = dim_in
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head
        self.dim_out = dim_out
        self.int8 = int8
        self.length_scale = length_scale
        self.attn_scale = attn_scale
        self.mask_value = mask_value
        self.dtype = dtype
        self.dropout_p = dropout_p
        self.shared_key_and_value = shared_key_and_value

        if dropout_p:
            self.attention_dropout = torch.nn.Dropout(dropout_p)
        else:
            self.attention_dropout = None
        
        self.bias = bias
        self.pos_bias_type = pos_bias_type
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query : torch.Tensor,
                      key_value : torch.Tensor,
                      attention_mask : torch.Tensor,
                      position_bias : Optional[torch.Tensor] = None,
                      use_cache: bool = False,
                      past_key_value = None,
        ):

        """ This model inherits from bmt.DistributedModule. 

        Args:
            query (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            key_value (:obj:`torch.Tensor` of shape ``(batch, len_k, dim_model)``): Length of input sequence before padding.  
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, len_q, len_k)``): Used to avoid performing attention on padding token indices.
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, len_q, len_k)`` or ``(1, num_heads, len_k, len_q)``): Provide positional information about tensor `key_value` and `query`. 

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): The attention output.
        """

        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        h_q = self.project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = self.project_k(key_value)         # (batch, len_k, num_heads * dim_head)
        h_v = self.project_v(key_value)         # (batch, len_k, num_heads * dim_head)

        h_q = h_q.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1, 3)   # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, self.num_heads_kv, self.dim_head).permute(0, 2, 1, 3)   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, self.num_heads_kv, self.dim_head).permute(0, 2, 1, 3)   # (batch, num_heads_kv, len_k, dim_head)

        # if self.shared_key_and_value:
        #     h_k = h_k.repeat(1, self.num_heads, 1, 1)
        #     h_v = h_v.repeat(1, self.num_heads, 1, 1)

        h_q = h_q.contiguous()      # (batch * num_heads, len_q, dim_head)
        h_k = h_k.contiguous()      # (batch * num_heads, len_k, dim_head)
        h_v = h_v.contiguous()      # (batch * num_heads, len_k, dim_head)

        if past_key_value is not None:
            h_k = torch.cat([past_key_value[0], h_k], dim=-2)
            h_v = torch.cat([past_key_value[1], h_v], dim=-2)
            len_k = h_k.size(-2)

        current_key_value = (h_k, h_v) if use_cache else None

        if self.pos_bias_type == "rotary":
            h_q, h_k = position_bias(h_q, h_k)

        # (batch, num_heads, len_q, dim_head) @ (batch, num_heads_kv, len_k, dim_head)T 
        # => (batch, num_heads, len_q, len_k)
        
        score = torch.matmul(h_q, h_k.transpose(2, 3))
        if self.attn_scale:
            score = score / math.sqrt(self.dim_head)

        # (batch, num_heads, len_q, len_k) 
        # score = score.view(batch_size, self.num_heads, len_q, len_k)

        if self.pos_bias_type == "relative":
            if position_bias is not None:
                # (batch, num_heads, len_q, len_k) + (1, num_heads, len_q, len_k) 
                score = score + position_bias
        
        score = torch.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k)==False,
            torch.scalar_tensor(self.mask_value, device=score.device, dtype=score.dtype)
        )   # (batch, num_heads, len_q, len_k)

        score = self.softmax(score)

        # avoid nan in softmax
        score = torch.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k)==False,
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype)
        )
        #.view(batch_size * self.num_heads, len_q, len_k) # (batch * num_heads, len_q, len_k)

        if self.attention_dropout is not None:
            score = self.attention_dropout(score)

         # (batch * num_heads, len_q, len_k) @ (batch * num_heads, len_k, dim_head) = (batch * num_heads, len_q, dim_head)
        score = torch.matmul(score, h_v)

        score = score.view(batch_size, self.num_heads, len_q, self.dim_head).permute(0, 2, 1, 3) # (batch, len_q, num_heads, dim_head)
        score = score.reshape(batch_size, len_q, self.num_heads * self.dim_head) # (batch, len_q, num_heads * dim_head)

        # (1#batch, dim_model, num_heads * dim_head) @ (batch, num_heads * dim_head, len_q) = (batch, dim_model, len_q)
        score = self.attention_out(score)

        if use_cache:
            return score, current_key_value
        else:
            return score


class SparseSelfAttention(Attention):

    def __init__(self, attention_window : int = 512, **kwargs):

        super().__init__(**kwargs)

        # separate projection layers for tokens with global attention
        self.project_q_global = Linear(
            dim_in = self.dim_in,
            dim_out = self.dim_out,
            length_scale=self.length_scale,
            length_scale_before=False,
            dtype = self.dtype,
            int8=self.int8,
            init_mean=self.init_mean,
            init_std=self.init_std,
            bias=self.bias,
        )

        self.project_k_global = Linear(
            dim_in = self.dim_in,
            dim_out = self.dim_out,
            length_scale=self.length_scale,
            length_scale_before=False,
            dtype = self.dtype,
            int8=self.int8,
            init_mean=self.init_mean,
            init_std=self.init_std,
            bias=self.bias,
        )

        self.project_v_global = Linear(
            dim_in = self.dim_in,
            dim_out = self.dim_out,
            length_scale=self.length_scale,
            length_scale_before=False,
            dtype = self.dtype,
            int8=self.int8,
            init_mean=self.init_mean,
            init_std=self.init_std,
            bias=self.bias,
         )

        self.attention_window = attention_window 
        assert self.attention_window % 2 == 0, "attention_window must be even"
        self.one_sided_attn_window_size = attention_window // 2

    def forward(self, hidden_states: torch.Tensor,
                      attention_mask : Optional[torch.Tensor] = None,
                      position_bias : Optional[torch.Tensor] = None,
                      use_cache: bool = False,
                      past_key_value = None,
        ):
        """
        [`LongformerSelfAttention`] expects *len(hidden_states)* to be multiple of *attention_window*. Padding to
        *attention_window* happens in [`LongformerModel.forward`] to avoid redoing the padding on each layer.

        The *attention_mask* is changed in [`LongformerModel.forward`] from 0, 1, 2 to:

            - -10000: no attention
            - 0: local attention
            - +10000: global attention
        """
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        hidden_states = hidden_states.transpose(0, 1)
        # project hidden states
        query_vectors = self.project_q(hidden_states)
        key_vectors = self.project_k(hidden_states)
        value_vectors = self.project_v(hidden_states)
        query_vectors /= math.sqrt(self.dim_head)
        seq_len, batch_size, embed_dim = query_vectors.size()

        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.dim_head).transpose(0, 1)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.dim_head).transpose(0, 1)

        if self.pos_bias_type == "rotary" and position_bias is not None:
            query_vectors, key_vectors = position_bias(query_vectors, key_vectors)

        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )

        # values to pad for attention probs
        remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]

        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
            remove_from_windowed_attention_mask, -10000.0
        )
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        )

        # pad local attention probs
        attn_scores += diagonal_mask

        # compute local attention probs from global attention keys and contact over window dim
        if is_global_attn:
            # compute global attn indices required through out forward fn
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self._get_global_attn_indices(is_index_global_attn)
            # calculate global attn probs from global key

            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )
            # concat to local_attn_probs
            # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)

            # free memory
            del global_key_attn_scores

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=self.dtype)

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        attn_probs = attn_probs.type_as(attn_scores)

        # free memory
        del attn_scores

        # apply dropout
        if self.attention_dropout is not None:
            attn_probs = self.attention_dropout(attn_probs)

        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.dim_head).transpose(0, 1)

        # compute local attention output with global attention value and add
        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.dim_head), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()

        # compute value for global attention and overwrite to attention output
        if is_global_attn:
            global_attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
            )

            # get only non zero global attn output
            nonzero_global_attn_output = global_attn_output[
                is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
            ]

            # overwrite values with global attention
            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(
                len(is_local_index_global_attn_nonzero[0]), -1
            )
            # The attention weights for tokens with global attention are
            # just filler values, they were never used to compute the output.
            # Fill with 0 now, the correct values are in 'global_attn_probs'.
            attn_probs[is_index_global_attn_nonzero] = 0
        attn_output = self.attention_out(attn_output)
        outputs = attn_output.transpose(0, 1)

        return outputs


    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = F.pad(
            hidden_states_padded, padding
        )  # padding value is not important because it will be overwritten
        hidden_states_padded = hidden_states_padded.view(
            *hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2)
        )
        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        shift every row 1 step right, converting columns into diagonals.

        Example:

        ```python
        chunked_hidden_states: [
            0.4983,
            2.6918,
            -0.0071,
            1.0492,
            -1.8348,
            0.7672,
            0.2986,
            0.0285,
            -0.7584,
            0.4206,
            -0.0405,
            0.1599,
            2.0514,
            -1.1600,
            0.5372,
            0.2629,
        ]
        window_overlap = num_rows = 4
        ```

                     (pad & diagonalize) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000
                       0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,
                       -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()
        chunked_hidden_states = F.pad(
            chunked_hidden_states, (0, window_overlap + 1)
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, -1
        )  # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # total_num_heads x num_chunks x window_overlap*window_overlap
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""

        # non-overlapping chunks of size = 2w
        hidden_states = hidden_states.view(
            hidden_states.size(0),
            hidden_states.size(1) // (window_overlap * 2),
            window_overlap * 2,
            hidden_states.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(hidden_states.size())
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride = list(hidden_states.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8

    def _sliding_chunks_query_key_matmul(self, query: torch.Tensor, key: torch.Tensor, window_overlap: int):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
        overlap of size window_overlap
        """
        batch_size, seq_len, num_heads, dim_head = query.size()

        chunks_count = seq_len // window_overlap - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, dim_head)
        key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, dim_head)

        query = self._chunk(query, window_overlap)
        key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x dim_head
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x dim_head
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        # convert diagonals into columns
        diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        padding = (0, 0, 0, 1)
        diagonal_chunked_attention_scores = F.pad(diagonal_chunked_attention_scores, padding)
        diagonal_chunked_attention_scores = diagonal_chunked_attention_scores.view(
            *diagonal_chunked_attention_scores.size()[:-2], diagonal_chunked_attention_scores.size(-1), diagonal_chunked_attention_scores.size(-2)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.

        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap :
        ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        batch_size, seq_len, num_heads, dim_head = value.size()

        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size()[:3] == value.size()[:3]
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        # group batch_size and num_heads dimensions into one
        value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, dim_head)

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = F.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, dim_head)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, dim_head).transpose(1, 2)

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """compute global attn indices required throughout forward pass"""
        # helper variable
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)

        # max number of global attn indices in batch
        max_num_global_attn_indices = num_global_attn_indices.max()

        # indices of global attn
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)

        # helper variable
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)

        # location of the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)

        # location of the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)
        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _concat_with_global_key_attn_probs(
        self,
        key_vectors,
        query_vectors,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    ):
        batch_size = key_vectors.shape[0]

        # create only global key vectors
        key_vectors_only_global = key_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.dim_head
        )

        key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[is_index_global_attn_nonzero]

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = torch.einsum("blhd,bshd->blhs", (query_vectors, key_vectors_only_global))

        attn_probs_from_global_key[
            is_local_index_no_global_attn_nonzero[0], :, :, is_local_index_no_global_attn_nonzero[1]
        ] = -10000.0

        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(
        self,
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
    ):
        batch_size = attn_probs.shape[0]

        # cut local attn probs to global only
        attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
        # get value vectors for global only
        value_vectors_only_global = value_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.dim_head
        )
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[is_index_global_attn_nonzero]

        # use `matmul` because `einsum` crashes sometimes with fp16
        # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
        # compute attn output only global
        attn_output_only_global = torch.matmul(
            attn_probs_only_global.transpose(1, 2).clone(), value_vectors_only_global.transpose(1, 2).clone()
        ).transpose(1, 2)

        # reshape attn probs
        attn_probs_without_global = attn_probs.narrow(
            -1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices
        ).contiguous()

        # compute attn output with global
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors, self.one_sided_attn_window_size
        )
        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(
        self,
        hidden_states,
        max_num_global_attn_indices,
        is_local_index_global_attn_nonzero,
        is_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
        is_index_masked,
    ):
        seq_len, batch_size = hidden_states.shape[:2]

        # prepare global hidden states
        global_attn_hidden_states = hidden_states.new_zeros(max_num_global_attn_indices, batch_size, self.dim_out)
        global_attn_hidden_states[is_local_index_global_attn_nonzero[::-1]] = hidden_states[
            is_index_global_attn_nonzero[::-1]
        ]

        # global key, query, value
        global_query_vectors_only_global = self.project_q_global(global_attn_hidden_states)
        global_key_vectors = self.project_k_global(hidden_states)
        global_value_vectors = self.project_v_global(hidden_states)

        # normalize
        global_query_vectors_only_global /= math.sqrt(self.dim_head)

        # reshape
        global_query_vectors_only_global = (
            global_query_vectors_only_global.contiguous()
            .view(max_num_global_attn_indices, batch_size * self.num_heads, self.dim_head)
            .transpose(0, 1)
        )  # (batch_size * self.num_heads, max_num_global_attn_indices, dim_head)
        global_key_vectors = (
            global_key_vectors.contiguous().view(-1, batch_size * self.num_heads, self.dim_head).transpose(0, 1)
        )  # batch_size * self.num_heads, seq_len, dim_head)
        global_value_vectors = (
            global_value_vectors.contiguous().view(-1, batch_size * self.num_heads, self.dim_head).transpose(0, 1)
        )  # batch_size * self.num_heads, seq_len, dim_head)

        # compute attn scores
        global_attn_scores = torch.bmm(global_query_vectors_only_global, global_key_vectors.transpose(1, 2))

        assert list(global_attn_scores.size()) == [
            batch_size * self.num_heads,
            max_num_global_attn_indices,
            seq_len,
        ], f"global_attn_scores have the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)}, but is {global_attn_scores.size()}."

        global_attn_scores = global_attn_scores.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)

        global_attn_scores[
            is_local_index_no_global_attn_nonzero[0], :, is_local_index_no_global_attn_nonzero[1], :
        ] = -10000.0

        global_attn_scores = global_attn_scores.masked_fill(
            is_index_masked[:, None, None, :],
            -10000.0,
        )

        global_attn_scores = global_attn_scores.view(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)

        # compute global attn probs
        global_attn_probs_float = self.softmax(global_attn_scores)

        # apply layer head masking
        if self.attention_dropout is not None:
            global_attn_probs_float = self.attention_dropout(global_attn_probs_float.type_as(global_attn_scores))

        # global attn output
        global_attn_output = torch.bmm(global_attn_probs_float, global_value_vectors)


        global_attn_probs_float = global_attn_probs_float.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_output = global_attn_output.view(
            batch_size, self.num_heads, max_num_global_attn_indices, self.dim_head
        )
        return global_attn_output, global_attn_probs_float
        if use_cache:
            return score, current_key_value
        else:
            return score
