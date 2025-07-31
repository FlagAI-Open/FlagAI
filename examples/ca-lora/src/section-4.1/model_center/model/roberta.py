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
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from ..layer import Encoder, Embedding, Linear, LayerNorm
from .basemodel import BaseModel, BaseModelOutputWithPooling
from .config import RobertaConfig


class RobertaPooler(nn.Module):
    def __init__(self, dim_model: int):
        super().__init__()
        self.dense = Linear(dim_model, dim_model, bias=True)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.FloatTensor):
        pooled_output = self.dense(hidden_states[:, 0, :])
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RobertaLMHead(nn.Module):
    def __init__(self, dim_model: int, vocab_size: int, norm_eps: float):
        super().__init__()
        self.dense = Linear(dim_model, dim_model, bias=True)
        self.activation = F.gelu
        self.layer_norm = LayerNorm(dim_model, eps=norm_eps)
        self.decoder = Linear(dim_model, vocab_size, bias=True)

    def forward(self, hidden_states: torch.FloatTensor, input_embedding: Optional[Embedding] = None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states) if input_embedding is None else \
                 input_embedding.projection(hidden_states) + self.decoder.bias
        return logits


class Roberta(BaseModel):

    _CONFIG_TYPE = RobertaConfig

    def __init__(self, config: RobertaConfig):

        super().__init__()
        # Model Config
        self.config = config
        # Embedding Layer
        self.input_embedding = Embedding(
            vocab_size=config.vocab_size,
            embedding_size=config.dim_model,
            length_scale=config.length_scale,
            dtype=config.dtype,
            int8=config.int8,
            init_mean=config.emb_init_mean,
            init_std=config.emb_init_std,
        )
        self.position_embedding = Embedding(
            vocab_size=config.position_size,
            embedding_size=config.dim_model,
            length_scale=config.length_scale,
            dtype=config.dtype,
            int8=config.int8,
            init_mean=config.emb_init_mean,
            init_std=config.emb_init_std,
        )
        self.token_type_embedding = Embedding(
            vocab_size=config.type_size,
            embedding_size=config.dim_model,
            length_scale=config.length_scale,
            dtype=config.dtype,
            int8=config.int8,
            init_mean=config.emb_init_mean,
            init_std=config.emb_init_std,
        )
        self.embed_dropout = nn.Dropout(config.dropout_p)
        # RoBERTa Model
        self.encoder = Encoder(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            dim_ff=config.dim_ff,
            num_heads=config.num_heads,
            dim_head=config.dim_head,
            dtype=config.dtype,
            int8=config.int8,
            norm_eps=config.norm_eps,
            norm_init_var=config.norm_init_var,
            norm_bias=config.norm_bias,
            att_init_mean=config.att_init_mean,
            att_init_std=config.att_init_std,
            att_bias=config.att_bias,
            att_mask_value=float(config.att_mask_value),
            pos_bias_type=config.pos_bias_type,
            ffn_init_mean=config.ffn_init_mean,
            ffn_init_std=config.ffn_init_std,
            ffn_bias=config.ffn_bias,
            ffn_activate_fn=config.ffn_activate_fn,
            length_scale=config.length_scale,
            attn_scale=config.attn_scale,
            dropout_p=config.dropout_p,
            post_layer_norm=config.post_layer_norm,
        )
        # Output Layer
        if config.cls_head:
            self.cls_projection = Linear(
                dim_out = config.cls_head,
                dim_in = config.dim_model,
                length_scale = config.length_scale,
                dtype = config.dtype,
                int8 = config.int8,
                init_mean = config.proj_init_mean,
                init_std = config.proj_init_std,
                bias = config.proj_bias,
            )
        self.lm_head = RobertaLMHead(
            dim_model = config.dim_model,
            vocab_size = config.vocab_size,
            norm_eps = config.norm_eps,
        )
        self.pooler = RobertaPooler(config.dim_model)
        self.padding_idx = config.pad_token_id

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                length: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = False,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                output_logits: Optional[bool] = False,
                output_pooler_output: Optional[bool] = False,
                output_attentions: Optional[bool] = False,
                output_hidden_states: Optional[bool] = False,
                return_dict: Optional[bool] = True,
        ):

        """ This model inherits from BaseModel. This model is also a PyTorch torch.nn.Module subclass.
            You can use it as a regular PyTorch Module. You can also select the data and data type that 
            you want the model to return through changing the value of `output_logits`, 
            `output_pooler_output`, `output_attentions`, `output_hidden_states` and `return_dict`.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.
            length (`torch.LongTensor` of shape `(batch_size)`, *optional*):
                Length of input sequence before padding.  
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. The values are selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                At least one of `length` and `attention_mask` must be given.
            token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Segment token indices to indicate first and second portions of the inputs. The values are selected in `[0, 1]`:
                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. The values are selected in the range `[0,
                config.position_size - 1]`.
            head_mask (`torch.LongTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the self-attention modules. The values are selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
                Unused now.
            inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
                is useful if you want to convert `input_ids` indices into associated vectors rather than the model's internal 
                token vectors.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
            past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.num_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, dim_model)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            output_logits (`bool`, *optional*):
                Whether or not to return the prediction score for each token in vocabulary (before softmax).
            output_pooler_output (`bool`, *optional*):
                Whether or not to return the pooler output of the last layer.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. 
                Unused now.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
                Unused now.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`BaseModelOutputWithPooling`] instead of a plain tuple.

        Return:
            BaseModelOutputWithPooling or tuple: The RoBERTa output. 
            Depended on the value of `output_logits`, `output_pooler_output`, `output_attentions`, 
            `output_hidden_states` and `return_dict`.
        """

        # encode the input into embeddings.
        with torch.no_grad():
            assert input_ids is not None or inputs_embeds is not None
            if input_ids is not None:
                batch = input_ids.size(0)
                input_length = input_ids.size(1)
                device = input_ids.device
            else:
                batch = inputs_embeds.size(0)
                input_length = inputs_embeds.size(1)
                device = inputs_embeds.device
            pkv_len = 0 if past_key_values is None else past_key_values[0][0].size(-2)
            seq_length = pkv_len + input_length

            if attention_mask is None and length is None:
                length = torch.ones((batch,), dtype=torch.int32, device=device) * seq_length
            attention_mask = attention_mask.to(torch.bool) if attention_mask is not None else \
                torch.arange(seq_length, device=device)[None, :].repeat(batch, 1) < length[:, None]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.view(batch, seq_length, 1) & attention_mask.view(batch, 1, seq_length)
            attention_mask = attention_mask[:, -input_length:, :]

            if position_ids is None:
                position_ids = torch.arange(self.padding_idx + 1, seq_length + self.padding_idx + 1, dtype=torch.int32,
                                            device=device)[None, :].repeat(batch, 1)
            position_ids = position_ids[:, -input_length:]

            if token_type_ids is None:
                token_type_ids = torch.zeros(seq_length, dtype=torch.int32, device=device)[None, :].repeat(batch, 1)
            token_type_ids = token_type_ids[:, -input_length:]

        if inputs_embeds is None:
            last_hidden_state = self.input_embedding(input_ids)
        else:
            last_hidden_state = inputs_embeds
        position_embeds = self.position_embedding(position_ids)
        token_type_embeds = self.token_type_embedding(token_type_ids)

        last_hidden_state = last_hidden_state + token_type_embeds + position_embeds
        last_hidden_state = self.embed_dropout(last_hidden_state)

        # input the input embeddings into the RoBERTa model
        current_key_values = None
        if use_cache:
            last_hidden_state, current_key_values = self.encoder(last_hidden_state, attention_mask, 
                                                             use_cache = use_cache, past_key_values = past_key_values)
        else:
            last_hidden_state = self.encoder(last_hidden_state, attention_mask)

        # use the hidden states of the last layer for sequential tasks, such as sequential labeling and language modeling.
        logits = None
        if output_logits:
            if self.config.cls_head:
                logits = self.cls_projection(last_hidden_state)
            elif self.config.tied:
                logits = self.lm_head(last_hidden_state, self.input_embedding)
            elif not self.config.tied:
                logits = self.lm_head(last_hidden_state)

        # use the hidden state of the first token for classification task.
        pooler_output = self.pooler(last_hidden_state) if output_pooler_output else None

        # BaseModelOutputWithPooling or tuple: The RoBERTa output. 
        if not return_dict:
            return last_hidden_state, pooler_output, current_key_values, logits, None, None
        else:
            return BaseModelOutputWithPooling(
                last_hidden_state = last_hidden_state,
                pooler_output = pooler_output,
                past_key_values = current_key_values,
                logits = logits,
                hidden_states = None,
                attentions = None,
            )