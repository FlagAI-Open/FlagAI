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
import torch.nn.functional as F
from ..layer import Encoder, Embedding, Linear, LayerNorm
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from .basemodel import BaseModel
from .config import LongformerConfig
class LongformerPooler(torch.nn.Module):
    def __init__(self, dim_model):
        super().__init__()
        self.dense = Linear(dim_model, dim_model, bias=True)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states[:, 0, :])
        pooled_output = self.activation(pooled_output)
        return pooled_output

        
class LongformerLMHead(torch.nn.Module):
    def __init__(self, dim_model, vocab_size, norm_eps, dtype):
        super().__init__()
        self.dense = Linear(dim_model, dim_model, bias=True, dtype = dtype)
        self.act_fn = torch.nn.functional.gelu
        self.layer_norm = LayerNorm(dim_model, eps=norm_eps, dtype = dtype)
        self.decoder = Linear(dim_model, vocab_size, bias=True, dtype = dtype)

    def forward(self, hidden_states, input_embedding):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = input_embedding.projection(hidden_states) + self.decoder.bias

        return logits


class Longformer(BaseModel):

    _CONFIG_TYPE = LongformerConfig

    def __init__(self, config: LongformerConfig):
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.input_embedding = Embedding(
            vocab_size = config.vocab_size,
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
            padding_idx=config.pad_token_id,
        )

        self.position_embedding = Embedding(
            vocab_size = config.position_size, 
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
            padding_idx=config.pad_token_id,
        )

        self.token_type_embedding = Embedding(
            vocab_size = config.type_size,
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )
        self.dtype = config.dtype
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
            post_layer_norm = config.post_layer_norm,
            sparse_attention = True,
            attention_window = config.attention_window,
        )

        self.tied = config.tied
        self.cls_head = config.cls_head
        self.attention_window = config.attention_window
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        if self.cls_head:
            self.cls_projection = Linear(
                dim_out = self.cls_head,
                dim_in = config.dim_model,
                length_scale = config.length_scale,
                dtype = config.dtype,
                int8 = config.int8,
                init_mean = config.proj_init_mean,
                init_std = config.proj_init_std,
                bias = config.proj_bias,
            )
        self.lm_head = LongformerLMHead(
            dim_model = config.dim_model,
            vocab_size = config.vocab_size,
            norm_eps = config.norm_eps,
            dtype = config.dtype
        )

        self.pooler = LongformerPooler(config.dim_model)
    def _pad_to_window_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        pad_token_id: int,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.attention_window
            if isinstance(self.attention_window, int)
            else max(self.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            if input_ids is not None:
                input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
                position_ids = F.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full((batch_size, padding_len), self.pad_token_id, dtype=torch.long,)
                inputs_embeds_padding = self.input_embedding(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            attention_mask = F.pad(attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens
            token_type_ids = F.pad(token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0

        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds
    def forward(self,
                input_ids=None,
                length=None,
                attention_mask=None,
                global_attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                return_dict=True,
                return_logits = False,
    ):
        """ This model inherits from BaseModel. This model is also a PyTorch torch.nn.Module subclass.
            You can use it as a regular PyTorch Module.
            You can also select the data and data type that you want the model to return through changing the value of `return_dict` and `return_logits`.

        Args:
            input_ids (:obj:`torch.Tensor` of shape ``(batch, seq_length)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            length (:obj:`torch.Tensor` of shape ``(batch)``): Length of input sequence before padding.  
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_length)``): Used to avoid performing attention on padding token indices.
            token_type_ids(:obj:`torch.Tensor` of shape ``(batch, seq_length)``): Unused. 
            position_ids(:obj:`torch.Tensor` of shape ``(batch, seq_length)``): Unused.
            head_mask (:obj:`torch.Tensor` of shape ``(num_layers, num_heads)``): Unused.
            inputs_embeds (:obj:`torch.Tensor` of shape ``(batch, seq_length, dim_model)``): Embedding of the input. You can choose to directly pass the inputs embedding to control the way of embedding. 
            encoder_hidden_states(:obj:`torch.Tensor` of shape(batch, seq_length, dim_model)): Unused.
            encoder_attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_length)``): Unused. 
            output_attentions (:obj:`torch.Tensor` of shape ``(batch, num_heads, seq_length, seq_length)``): Unused.
            output_hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_length, dim_model)``): Unused.
            return_dict (:obj:`bool`): Whether to return a BaseModelOutputWithPoolingAndCrossAttentions instead of just a tuple.
            return_logits (:obj:`bool`): Whether to return the prediction score for each token in vocabulary (before softmax).

        Return:
            BaseModelOutputWithPoolingAndCrossAttentions or tuple or torch.Tensor of shape (batch, seq_length, vocab_output_size) or (batch, seqlen, cls_head): The Bert output. Depended on the value of `return_dict` and `return_logits` 

        """
        assert input_ids is not None or inputs_embeds is not None

        if input_ids is not None:
            batch = input_ids.size(0)
            input_length = input_ids.size(1)
            device = input_ids.device
        else:
            batch = inputs_embeds.size(0)
            input_length = inputs_embeds.size(1)
            device = inputs_embeds.device
        padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.pad_token_id,
        )
        with torch.no_grad():

            if attention_mask is None:
                attention_mask = torch.ones(input_ids.size(),device=device).to(torch.bool)
            if global_attention_mask is not None:
                attention_mask = attention_mask * (global_attention_mask + 1)
                
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            if position_ids is None:
                if input_ids is not None:
                    mask = input_ids.ne(self.padding_idx).int()

                    position_ids = torch.cumsum(mask, dim=1).type_as(mask) * mask
                    position_ids = position_ids + self.padding_idx
                else:
                    input_shape = inputs_embeds.size()[:-1]
                    position_ids = torch.arange(
                        self.padding_idx + 1, input_length + self.padding_idx + 1, dtype=torch.int32, device=inputs_embeds.device
                    ).unsqueeze(0).expand(input_shape)

            if token_type_ids is None:
                token_type_ids = torch.zeros(input_length, dtype=torch.int32, device=device)[None, :].repeat(batch, 1)

        attention_mask = attention_mask.to(torch.int32)-1
        # the longformer author says it will avoid fp16 overflow or underflow
        if inputs_embeds is None:
            hidden_states = self.input_embedding(input_ids.to(torch.int32))
        else:
            hidden_states = inputs_embeds
        position_embeds = self.position_embedding(position_ids.to(torch.int32))
        token_type_embeds = self.token_type_embedding(token_type_ids.to(torch.int32))
        hidden_states = hidden_states + token_type_embeds + position_embeds

        hidden_states = self.encoder(hidden_states, attention_mask)

        if self.cls_head:
            logits = self.cls_projection(hidden_states)
        logits = self.lm_head(hidden_states, self.input_embedding)

        if return_logits:
            return logits

        pooled_output = self.pooler(hidden_states)

        if not return_dict:
            return (hidden_states, pooled_output, None, None, None, None)
        else:
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=hidden_states,
                pooler_output=pooled_output,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                cross_attentions=None,
            )