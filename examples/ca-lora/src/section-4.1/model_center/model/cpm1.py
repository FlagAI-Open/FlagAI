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
from typing import Optional, List
from ..layer import Encoder, Embedding, Linear, RelativePositionEmbedding
from .basemodel import BaseModel, BaseModelOutput
from .config import CPM1Config


class CPM1(BaseModel):

    _CONFIG_TYPE = CPM1Config

    def __init__(self, config: CPM1Config):

        super().__init__()
        # Model Config
        self.config = config
        # Embedding Layer
        self.input_embedding = Embedding(
            vocab_size = config.vocab_size, 
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )
        self.position_bias = RelativePositionEmbedding(
            num_heads = config.num_heads, 
            num_buckets = config.position_bias_num_buckets, 
            max_distance = config.position_bias_max_distance, 
            bidirectional = True,
            dtype = config.dtype,
            init_mean = config.pos_init_mean,
            init_std = config.pos_init_std,
        )
        # CPM-1 Model
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
        if not config.tied:
            self.output_projection = Linear(
                dim_out = config.vocab_size,
                dim_in = config.dim_model,
                length_scale = config.length_scale,
                dtype = config.dtype,
                int8 = config.int8,
                init_mean = config.proj_init_mean,
                init_std = config.proj_init_std,
                bias = config.proj_bias,
            )

    def forward(self, 
                input_ids: Optional[torch.Tensor] = None,
                length: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                span: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = False,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                output_logits: Optional[bool] = False,
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
                Segment token indices to indicate first and second portions of the inputs. 
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. 
            context (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Value to show whether tokens are contexts or not.
                - Ture for tokens that are contexts
                - False for tokens that are not contexts
            span (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence spans. The attention between the tokens in different spans is set to 
                negative infinity. This can bundle multiple inputs for processing.
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
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. 
                Unused now.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
                Unused now.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`BaseModelOutput`] instead of a plain tuple.

        Return:
            BaseModelOutput or tuple: The CPM-1 output. 
            Depended on the value of `output_logits`, `output_attentions`, `output_hidden_states` and `return_dict`.
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

            attention_mask = attention_mask.to(torch.bool) if attention_mask is not None else \
                torch.arange(seq_length, device=device)[None, :].repeat(batch, 1) < length[:, None]
            if attention_mask.dim() == 2:
                directional_mask_2d = torch.arange(seq_length, device=device) <= torch.arange(seq_length, device=device).view(-1, 1)
                if context is None:
                    attention_mask = attention_mask.view(batch, 1, seq_length) & directional_mask_2d.view(1, seq_length, seq_length)
                else:
                    attention_mask = attention_mask.view(batch, seq_length, 1) & attention_mask.view(batch, 1, seq_length) & \
                        (context[:, None, :] | (context[:, :, None].logical_not() & directional_mask_2d.view(1, seq_length, seq_length)))
            if span is not None:
                attention_mask = (span[:, None, :] == span[:, :, None]) & attention_mask
            attention_mask = attention_mask[:, -input_length:, :]

        position_bias = self.position_bias(position_ids, position_ids) if position_ids is not None else \
            self.position_bias(seq_length, seq_length)
        position_bias = position_bias[:, -input_length:, :]
        hidden_states = self.input_embedding(input_ids) if inputs_embeds is None else inputs_embeds

        # input the input embeddings into the CPM-1 model
        current_key_values = None
        if use_cache:
            hidden_states, current_key_values = self.encoder(hidden_states, attention_mask, position_bias,
                                                             use_cache = use_cache, past_key_values = past_key_values)
        else:
            hidden_states = self.encoder(hidden_states, attention_mask, position_bias)

        # use the hidden states of the last layer for sequential tasks, such as sequential labeling and language modeling.
        logits = None
        if output_logits:
            if self.config.cls_head:
                logits = self.cls_projection(hidden_states)
            elif self.config.tied:
                logits = self.input_embedding.projection(hidden_states)
            elif not self.config.tied:
                logits = self.output_projection(hidden_states)

        # BaseModelOutput or tuple: The CPM-1 output. 
        if not return_dict:
            return hidden_states, current_key_values, logits, None, None
        else:
            return BaseModelOutput(
                last_hidden_state = hidden_states,
                past_key_values = current_key_values,
                logits = logits,
                hidden_states = None,
                attentions = None,
            )