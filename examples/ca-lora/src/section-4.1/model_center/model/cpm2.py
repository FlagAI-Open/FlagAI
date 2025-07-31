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
from ..layer import Encoder, Decoder, Embedding, Linear, RelativePositionEmbedding
from .basemodel import BaseModel, Seq2SeqModelOutput
from .config import CPM2Config


class CPM2(BaseModel):

    _CONFIG_TYPE = CPM2Config

    def __init__(self, config: CPM2Config):

        super().__init__()
        # Model Config
        self.config = config
        # Embedding Layer
        self.input_embedding = Embedding(
            vocab_size = config.vocab_size, 
            embedding_size = config.dim_model,
            length_scale = False, # TODO not an elegent implementation # config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )
        self.position_bias_enc = RelativePositionEmbedding(
            num_heads = config.num_heads, 
            num_buckets = config.position_bias_num_buckets, 
            max_distance = config.position_bias_max_distance, 
            bidirectional = True, 
            dtype = config.dtype,
            init_mean = config.pos_init_mean,
            init_std = config.pos_init_std,
        )
        self.position_bias_dec = RelativePositionEmbedding(
            num_heads = config.num_heads, 
            num_buckets = config.position_bias_num_buckets, 
            max_distance = config.position_bias_max_distance, 
            bidirectional = False, 
            dtype = config.dtype,
            init_mean = config.pos_init_mean,
            init_std = config.pos_init_std,
        )
        # CPM-2 Model
        self.encoder = Encoder(
            num_layers = config.num_encoder_layers,
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
        self.decoder = Decoder(
            num_layers = config.num_decoder_layers,
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
                decoder_input_ids: Optional[torch.Tensor] = None,
                decoder_length: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                decoder_head_mask: Optional[torch.Tensor] = None,
                cross_attn_head_mask: Optional[torch.Tensor] = None,
                encoder_outputs: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = False,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                output_attentions: Optional[bool] = False,
                output_hidden_states: Optional[bool] = False,
                output_decoder_attentions: Optional[bool] = False,
                output_decoder_hidden_states: Optional[bool] = False,
                output_logits: Optional[bool] = False,
                return_dict: Optional[bool] = True,
        ):
        """
            This model inherits from BaseModel. This model is also a PyTorch torch.nn.Module subclass.
            You can use it as a regular PyTorch Module. You can also select the data and data type that 
            you want the model to return through changing the value of `return_dict`, `output_attentions`, 
            `output_hidden_states`, `output_decoder_attentions`, `output_decoder_hidden_states` and `output_logits`.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of encoding sequence tokens in the vocabulary.
            length (`torch.LongTensor` of shape `(batch_size)`, *optional*):
                Length of encoding sequence before padding.  
            decoder_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of decoding sequence tokens in the vocabulary.
            decoder_length (`torch.LongTensor` of shape `(batch_size)`, *optional*):
                Length of decoding sequence before padding.  
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices in the encoding input. The values are selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                At least one of `length` and `attention_mask` must be given.
            decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices in the decoding input. The values are selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                At least one of `decoder_length` and `decoder_attention_mask` must be given.
            head_mask (`torch.LongTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the encoding self-attention modules. The values are selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
                Unused now.
            decoder_head_mask (`torch.LongTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the decoding self-attention modules. The values are selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
                Unused now.
            cross_attn_head_mask (`torch.LongTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the decoding cross-attention modules. The values are selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
                Unused now.
            encoder_outputs (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
                The hidden states of the last layer of the encoder.
            inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
                is useful if you want to convert `input_ids` indices into associated vectors rather than the model's internal 
                token vectors.
            decoder_inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
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
                Whether or not to return the attentions tensors of all attention layers  in the encoder. 
                Unused now.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers in the encoder.
                Unused now.
            output_decoder_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers in the decoder. 
                Unused now.
            output_decoder_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers in the decoder.
                Unused now.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`Seq2SeqModelOutput`] instead of a plain tuple.

        Return:
            Seq2SeqModelOutput or tuple: 
                The CPM-2 output. 
        """

        # encoder
        if encoder_outputs is None:
            with torch.no_grad():
                assert input_ids is not None or inputs_embeds is not None
                if input_ids is not None:
                    batch = input_ids.size(0)
                    seq_enc = input_ids.size(1)
                    device = input_ids.device
                else:
                    batch = inputs_embeds.size(0)
                    seq_enc = inputs_embeds.size(1)
                    device = inputs_embeds.device

                attention_mask = attention_mask.to(torch.bool) if attention_mask is not None else \
                    torch.arange(seq_enc, device=device)[None, :].repeat(batch, 1) < length[:, None]
                enc_attention_mask = attention_mask.view(batch, seq_enc, 1) & attention_mask.view(batch, 1, seq_enc)

            enc_position_bias = self.position_bias_enc(seq_enc, seq_enc)
            hidden_states_enc = self.input_embedding(input_ids) if inputs_embeds is None else inputs_embeds
            encoder_outputs = self.encoder(hidden_states_enc, enc_attention_mask, enc_position_bias)
        else:
            seq_enc = encoder_outputs.size(1)

        # decoder
        with torch.no_grad():
            assert decoder_input_ids is not None or decoder_inputs_embeds is not None
            if decoder_input_ids is not None:
                batch = decoder_input_ids.size(0)
                seq_dec = decoder_input_ids.size(1)
                device = decoder_input_ids.device
            else:
                batch = decoder_inputs_embeds.size(0)
                seq_dec = decoder_inputs_embeds.size(1)
                device = decoder_inputs_embeds.device
            pkv_len = 0 if past_key_values is None else past_key_values[0][0].size(-2)
            seq_length = pkv_len + seq_dec

            decoder_attention_mask = decoder_attention_mask.to(torch.bool) if decoder_attention_mask is not None else \
                torch.arange(seq_length, device=device)[None, :].repeat(batch, 1) < decoder_length[:, None]
            directional_mask_2d = torch.arange(seq_length, device=device) <= torch.arange(seq_length, device=device).view(-1, 1)
            dec_attention_mask = decoder_attention_mask.view(batch, seq_length, 1) & decoder_attention_mask.view(batch, 1, seq_length) & directional_mask_2d.view(1, seq_length, seq_length)
            dec_attention_mask = dec_attention_mask[:, -seq_dec:, :]
            cross_attention_mask = attention_mask.view(batch, 1, seq_enc) & decoder_attention_mask.view(batch, seq_length, 1)
            cross_attention_mask = cross_attention_mask[:,-seq_dec:,:]

        dec_position_bias = self.position_bias_dec(seq_length, seq_length)
        dec_position_bias = dec_position_bias[:, -seq_dec:, :]
        hidden_states_dec = self.input_embedding(decoder_input_ids) if decoder_inputs_embeds is None else decoder_inputs_embeds

        current_key_values = None
        if use_cache:
            decoder_outputs, current_key_values=self.decoder(hidden_states_dec, dec_attention_mask, dec_position_bias, 
                                                             encoder_outputs, cross_attention_mask, None,
                                                             use_cache = use_cache, past_key_values = past_key_values)
        else:
            decoder_outputs = self.decoder(hidden_states_dec, dec_attention_mask, dec_position_bias,
                                           encoder_outputs, cross_attention_mask, None)

        logits = None
        if output_logits:
            if self.config.cls_head:
                logits = self.cls_projection(decoder_outputs)
            elif self.config.tied:
                logits = self.input_embedding.projection(decoder_outputs)
            elif not self.config.tied:
                logits = self.output_projection(decoder_outputs)

        if not return_dict:
            return decoder_outputs, encoder_outputs, current_key_values, logits, None, None, None, None, None
        else:
            return Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs,
                encoder_last_hidden_state=encoder_outputs,
                past_key_values=current_key_values,
                logits = logits,
                encoder_hidden_states=None,
                decoder_hidden_states=None,
                decoder_attentions=None,
                cross_attentions=None,
                encoder_attentions=None,
            )