# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch T5 model. """

import copy
import warnings
import torch
import os
from torch import nn
from torch.nn import CrossEntropyLoss
from torch import Tensor, device
from typing import Tuple, Optional
from flagai.model.base_model import BaseModel
from flagai.model.blocks.t5_block import T5Block
from flagai.model.layers.layer_norm import T5LayerNorm
from flagai.model.layers.attentions import T5Attention
from flagai.model.layers.feedforward import T5DenseReluDense, T5DenseGatedGeluDense
from flagai.data.tokenizer.t5.t5_tokenizer import T5JiebaTokenizer
if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
    from flagai.mpu import copy_to_model_parallel_region
    from flagai.mpu.random import checkpoint
elif os.getenv('ENV_TYPE') == 'deepspeed':
    from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
else:
    from torch.utils.checkpoint import checkpoint
# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


class T5PreTrainedModel(BaseModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # super(T5PreTrainedModel, self).__init__(config, **kwargs)

    def init_weights(self):
        for module in self.modules():
            self._init_weights(module)

    def _init_weights(self, module):
        """ Initialize the weights """
        factor = self.config[
            'initializer_factor']  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)

        elif isinstance(module, (T5Model)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0,
                                          std=factor *
                                          ((self.config['d_model'])**-0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0,
                                          std=factor *
                                          ((self.config['d_ff'])**-0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedGeluDense):
            module.wi_0.weight.data.normal_(mean=0.0,
                                            std=factor *
                                            ((self.config['d_model'])**-0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0,
                                            std=factor *
                                            ((self.config['d_model'])**-0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0,
                                          std=factor *
                                          ((self.config['d_ff'])**-0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config['d_model']
            key_value_proj_dim = self.config['d_kv']
            n_heads = self.config['num_heads']
            module.q.weight.data.normal_(
                mean=0.0, std=factor * ((d_model * key_value_proj_dim)**-0.5))
            module.k.weight.data.normal_(mean=0.0,
                                         std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0,
                                         std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(
                mean=0.0, std=factor * ((n_heads * key_value_proj_dim)**-0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(
                    mean=0.0, std=factor * ((d_model)**-0.5))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config['decoder_start_token_id']
        pad_token_id = self.config['pad_token_id']

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(
        ), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class T5Stack(nn.Module):

    def __init__(self, config, embed_tokens=None):
        super().__init__()
        self.config = config
        self.embed_tokens = embed_tokens
        self.is_decoder = config['is_decoder']
        self.block = nn.ModuleList([
            T5Block(config, has_relative_attention_bias=bool(i == 0))
            for i in range(config['num_layers'])
        ])
        self.final_layer_norm = T5LayerNorm(
            self.config['d_model'], eps=self.config['layer_norm_epsilon'])
        self.dropout = nn.Dropout(self.config['dropout_rate'])

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.dtype = self.embed_tokens.weight.dtype

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds"
            )

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[
            2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(
                self)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(
                inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size,
                                                encoder_seq_length,
                                                device=inputs_embeds.device,
                                                dtype=torch.long)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config['num_layers'])
        encoder_head_mask = self.get_head_mask(encoder_head_mask,
                                               self.config['num_layers'])
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions
                                      and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module,
                past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            encoder_layer_head_mask = encoder_head_mask[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            if self.config['checkpoint_activations']:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    encoder_layer_head_mask,
                    past_key_value,
                    None,
                )
            else:

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    encoder_layer_head_mask=encoder_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (
                    present_key_value_state, )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3], )
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (
                        layer_outputs[5], )

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                present_key_value_states,
                all_hidden_states,
                all_attentions,
                all_cross_attentions,
            ] if v is not None)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": present_key_value_states,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "cross_attentions": all_cross_attentions,
        }

    def get_head_mask(self,
                      head_mask: Optional[Tensor],
                      num_hidden_layers: int,
                      is_attention_chunked: bool = False) -> Tensor:
        """
        Prepare the head mask if needed.
        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.
        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask,
                                                      num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def get_extended_attention_mask(self, attention_mask: Tensor,
                                    input_shape: Tuple[int],
                                    device: device) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.
        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config['is_decoder']:
                extended_attention_mask = self.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device)
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def create_extended_attention_mask_for_decoder(self, input_shape,
                                                   attention_mask, device):
        batch_size, seq_length = input_shape
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = seq_ids[None, None, :].repeat(
            batch_size, seq_length, 1) <= seq_ids[None, :, None]
        # in case past_key_values are used we need to add a prefix ones mask to the causal mask
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.to(attention_mask.dtype)

        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat(
                [
                    torch.ones((batch_size, seq_length, prefix_seq_len),
                               device=device,
                               dtype=causal_mask.dtype),
                    causal_mask,
                ],
                axis=-1,
            )

        extended_attention_mask = causal_mask[:,
                                              None, :, :] * attention_mask[:,
                                                                           None,
                                                                           None, :]
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).
        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.
        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:,
                                                                     None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None,
                                                                     None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(
            dtype=self.dtype)  # fp16 compatibility

        if self.dtype == torch.float16:
            encoder_extended_attention_mask = (
                1.0 - encoder_extended_attention_mask) * -1e4
        elif self.dtype in [torch.bfloat16, torch.float32]:
            encoder_extended_attention_mask = (
                1.0 - encoder_extended_attention_mask) * -1e9
        else:
            raise ValueError(
                f"{self.dtype} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`"
            )

        return encoder_extended_attention_mask


class T5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.model_dim = config['d_model']
        self.shared = nn.Embedding(config['vocab_size'], config['d_model'])

        encoder_config = copy.deepcopy(config)
        encoder_config['is_decoder'] = False
        encoder_config['use_cache'] = False
        encoder_config['is_encoder_decoder'] = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config['is_decoder'] = True
        decoder_config['is_encoder_decoder'] = False
        decoder_config['num_layers'] = config['num_decoder_layers']
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config['d_model'],
                                 config['vocab_size'],
                                 bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config['decoder_start_token_id']
        pad_token_id = self.config['pad_token_id']

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(
        ), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids

    def forward(self, **data):
        input_ids = data.get("input_ids", None)
        attention_mask = data.get("attention_mask", None)
        decoder_input_ids = data.get("decoder_input_ids", None)
        decoder_attention_mask = data.get("decoder_attention_mask", None)
        head_mask = data.get("head_mask", None)
        decoder_head_mask = data.get("decoder_head_mask", None)
        encoder_outputs = data.get("encoder_outputs", None)
        past_key_values = data.get("past_key_values", None)
        inputs_embeds = data.get("inputs_embeds", None)
        decoder_inputs_embeds = data.get("decoder_inputs_embeds", None)
        labels = data.get("labels", None)
        use_cache = data.get("use_cache", None)
        output_attentions = data.get("output_attentions", None)
        output_hidden_states = data.get("output_hidden_states", None)
        return_dict = data.get("return_dict", None)

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config['num_layers'] == self.config['num_decoder_layers']:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        encoder_outputs = {
            "last_hidden_state": encoder_outputs[0],
            "hidden_states":
            encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            "attentions":
            encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        }

        hidden_states = encoder_outputs["last_hidden_state"]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        decoder_outputs = {
            "last_hidden_state": decoder_outputs[0],
            "hidden_states":
            decoder_outputs[1] if len(decoder_outputs) > 1 else None,
            "attentions":
            decoder_outputs[2] if len(decoder_outputs) > 2 else None,
        }
        sequence_output = decoder_outputs["last_hidden_state"]
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)),
                            labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        return {
            "loss": loss,
            "logits": lm_logits,
            "past_key_values": decoder_outputs.get("past_key_values", None),
            "decoder_hidden_states": decoder_outputs["hidden_states"],
            "decoder_attentions": decoder_outputs["attentions"],
            "cross_attentions": decoder_outputs.get("cross_attentions", None),
            "encoder_last_hidden_state": encoder_outputs["last_hidden_state"],
            "encoder_hidden_states": encoder_outputs["hidden_states"],
            "encoder_attentions": encoder_outputs["attentions"]
        }

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past=None,
                                      attention_mask=None,
                                      use_cache=None,
                                      encoder_outputs=None,
                                      **kwargs):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            print(
                "You might want to consider setting `use_cache=True` to speed up decoding"
            )
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx), )

            assert reordered_layer_past_states[0].shape == layer_past_states[
                0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (
                reordered_layer_past_states, )
        return reordered_decoder_past

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path,
                                map_location=torch.device("cpu"))
        if "module" in checkpoint:
            # ddp
            checkpoint = checkpoint["module"]
        self.load_state_dict(checkpoint, strict=True)
        return checkpoint


class T5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.model_dim = config['d_model']

        self.shared = nn.Embedding(config['vocab_size'], config['d_model'])

        encoder_config = copy.deepcopy(config)
        encoder_config['is_decoder'] = False
        encoder_config['use_cache'] = False
        encoder_config['is_encoder_decoder'] = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config['is_decoder'] = True
        decoder_config['is_encoder_decoder'] = False
        if 'number_decoder_layers' in config:
            decoder_config['num_layers'] = config['num_decoder_layers']
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config['d_model'],
                                 config['vocab_size'],
                                 bias=False)
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared.weight.data

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.weight.data = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head.weight

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        mems=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config['num_layers'] == self.config['num_decoder_layers']:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, dict):
            encoder_outputs = {
                "last_hidden_state":
                encoder_outputs[0],
                "hidden_states":
                encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                "attentions":
                encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            }

        hidden_states = encoder_outputs['last_hidden_state']

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(
                    self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs['last_hidden_state']

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        #
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)),
                            labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits, ) + decoder_outputs[1:] + encoder_outputs
            return ((loss, ) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits": lm_logits,
            "past_key_values": decoder_outputs["past_key_values"],
            "decoder_hidden_states": decoder_outputs["hidden_states"],
            "decoder_attentions": decoder_outputs["attentions"],
            "cross_attentions": decoder_outputs["cross_attentions"],
            "encoder_last_hidden_state": encoder_outputs["last_hidden_state"],
            "encoder_hidden_states": encoder_outputs["hidden_states"],
            "encoder_attentions": encoder_outputs["attentions"]
        }

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past=None,
                                      attention_mask=None,
                                      use_cache=None,
                                      encoder_outputs=None,
                                      **kwargs):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            print(
                "You might want to consider setting `use_cache=True` to speed up decoding"
            )
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx), )

            assert reordered_layer_past_states[0].shape == layer_past_states[
                0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (
                reordered_layer_past_states, )
        return reordered_decoder_past

    def load_weights(self, checkpoint_path):

        _keys_to_ignore_on_load_missing = [
            r"encoder\.embed_tokens\.weight",
            r"decoder\.embed_tokens\.weight",
            r"lm_head\.weight",
        ]
        _keys_to_ignore_on_load_unexpected = [
            r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
        ]

        checkpoint = torch.load(checkpoint_path,
                                map_location=torch.device("cpu"))
        if "module" in checkpoint:
            # ddp
            checkpoint = checkpoint["module"]
        self.load_state_dict(checkpoint, strict=False)
        self.set_output_embeddings(nn.Parameter(self.get_input_embeddings()))

        return checkpoint


class T5EncoderModel(T5PreTrainedModel):
    authorized_missing_keys = [
        r"encoder\.embed_tokens\.weight",
    ]

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config['vocab_size'], config['d_model'])

        encoder_config = copy.deepcopy(config)
        encoder_config['use_cache'] = False
        encoder_config['is_encoder_decoder'] = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        return_dict = return_dict if return_dict is not None else self.config[
            'use_return_dict']

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs


class T5Config():
    r"""
    This is the configuration class to store the configuration of a [`T5Model`] or a [`TFT5Model`]. It is used to
    instantiate a T5 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the T5
    [t5-small](https://huggingface.co/t5-small) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Arguments:
        vocab_size (`int`, *optional*, defaults to 32128):
            Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`T5Model`] or [`TFT5Model`].
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model //
            num_heads`.
        d_ff (`int`, *optional*, defaults to 2048):
            Size of the intermediate feed forward layer in each `T5Block`.
        num_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_decoder_layers (`int`, *optional*):
            Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance of the longer sequences for the bucket separation.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            The ratio for all dropout layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (`string`, *optional*, defaults to `"relu"`):
            Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. T5v1.1 uses the
            `"gated-gelu"` feed forward projection. Original T5 uses `"relu"`.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """

    def __init__(self,
                 vocab_size=32128,
                 d_model=512,
                 d_kv=64,
                 d_ff=2048,
                 num_layers=6,
                 num_decoder_layers=None,
                 num_heads=8,
                 relative_attention_num_buckets=32,
                 relative_attention_max_distance=128,
                 dropout_rate=0.1,
                 layer_norm_epsilon=1e-6,
                 initializer_factor=1.0,
                 feed_forward_proj="relu",
                 use_cache=True,
                 architectures=None,
                 decoder_start_token_id=None,
                 eos_token_id=None,
                 is_encoder_decoder=True,
                 model_type=None,
                 output_past=None,
                 pad_token_id=0,
                 tie_word_embeddings=None,
                 tokenizer_class=None,
                 **kwargs):
        self.tokenizer_class = tokenizer_class
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.output_past = output_past
        self.model_type = model_type
        self.is_encoder_decoder = is_encoder_decoder
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.architectures = architectures
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (num_decoder_layers if num_decoder_layers
                                   is not None else self.num_layers
                                   )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
        self.use_return_dict = True
        self.output_attentions = True
        self.output_hidden_states = True
        self.only_hidden_states = False
        self.return_logist_only = False


class T5UERConfig:

    model_type = "t5"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self,
                 vocab_size=50000,
                 d_model=768,
                 d_kv=64,
                 d_ff=2048,
                 num_layers=12,
                 num_decoder_layers=12,
                 num_heads=12,
                 relative_attention_num_buckets=32,
                 relative_attention_max_distance=128,
                 dropout_rate=0.1,
                 layer_norm_epsilon=1e-6,
                 initializer_factor=1.0,
                 feed_forward_proj="gated-gelu",
                 is_encoder_decoder=True,
                 use_cache=True,
                 pad_token_id=0,
                 eos_token_id=1,
                 is_decoder=False,
                 **kwargs):
        self.is_decoder = is_decoder
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (num_decoder_layers if num_decoder_layers
                                   is not None else self.num_layers
                                   )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
        self.use_return_dict = True
        self.output_attentions = True
        self.output_hidden_states = True
        self.only_hidden_states = True
        self.return_logist_only = True
        self.tie_word_embeddings = False

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.num_heads

    @property
    def num_hidden_layers(self):
        return self.num_layers


class T5SmallUERConfig:

    model_type = "t5"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self,
                 vocab_size=21228,
                 d_model=512,
                 d_kv=64,
                 d_ff=1024,
                 num_layers=8,
                 num_decoder_layers=8,
                 num_heads=6,
                 relative_attention_num_buckets=32,
                 relative_attention_max_distance=128,
                 dropout_rate=0.1,
                 layer_norm_epsilon=1e-6,
                 initializer_factor=1.0,
                 feed_forward_proj="gated-gelu",
                 is_encoder_decoder=True,
                 use_cache=True,
                 pad_token_id=0,
                 eos_token_id=1,
                 is_decoder=False,
                 **kwargs):
        self.is_decoder = is_decoder
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (num_decoder_layers if num_decoder_layers
                                   is not None else self.num_layers
                                   )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
        self.use_return_dict = True
        self.output_attentions = True
        self.output_hidden_states = True
        self.only_hidden_states = True
        self.return_logist_only = True
        self.tie_word_embeddings = False

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.num_heads

    @property
    def num_hidden_layers(self):
        return self.num_layers


class T5UERModel(nn.Module):

    def __init__(self, word2idx, size="base"):
        super().__init__()
        self.device = torch.device("cpu")
        if size == "base":
            config = T5UERConfig()
        elif size == "small":
            config = T5SmallUERConfig()
        else:
            raise Exception("not support this model type")

        self.model = T5ForConditionalGeneration(config)

        self.word2idx = word2idx
        self.tokenizer = T5JiebaTokenizer(self.word2idx)
        self.bos_id = self.word2idx["[CLS]"]
        self.eos_id = self.word2idx["[SEP]"]
        self.unk_id = self.word2idx["[UNK]"]

    def load_pretrain_params(self, pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)
        checkpoint = {"model." + k: v for k, v in checkpoint.items()}

        self.load_state_dict(checkpoint, strict=True)
        torch.cuda.empty_cache()

    def load_all_params(self, model_path, device="cuda"):
        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()

    def set_device(self, device):
        self.device = torch.device(device)
        self.to(device)

    def save_all_params(self, save_path):
        torch.save(self.state_dict(), save_path)

    def forward(self,
                input_ids,
                decoder_input_ids,
                labels=None,
                mems=None,
                checkpoint_fn=None):
        return self.model(input_ids=input_ids,
                          decoder_input_ids=decoder_input_ids,
                          labels=labels)


if __name__ == '__main__':
    T5Model.from_pretrain('./sss')
